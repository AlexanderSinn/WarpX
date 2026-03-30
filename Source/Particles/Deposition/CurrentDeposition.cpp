#include "Particles/Deposition/CurrentDeposition.H"



struct shape_factor_result {
    amrex::Real factor;
    int cell;
};

inline static constexpr amrex::GpuTuple<
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 1>, 1>,
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 2>, 2>,
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 3>, 3>,
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 4>, 4>,
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 5>, 5>
> depos_prefactor {
    {{
        {       1.},
    }}, {{
        {      -1.,       1.},
        {       1.,       0.}
    }}, {{
        {    1./2.,      -1.,    1./2.},
        {      -1.,       1.,    1./2.},
        {    1./2.,       0.,       0.}
    }}, {{
        {   -1./6.,    1./2.,   -1./2.,    1./6.},
        {    1./2.,      -1.,       0.,    2./3.},
        {   -1./2.,    1./2.,    1./2.,    1./6.},
        {    1./6.,       0.,       0.,       0.}
    }}, {{
        {   1./24.,   -1./6.,    1./4.,   -1./6.,   1./24.},
        {   -1./6.,    1./2.,   -1./4.,   -1./2.,  11./24.},
        {    1./4.,   -1./2.,   -1./4.,    1./2.,  11./24.},
        {   -1./6.,    1./6.,    1./4.,    1./6.,   1./24.},
        {   1./24.,       0.,       0.,       0.,       0.}
    }}
};

#ifdef AMREX_USE_CUDA
AMREX_GPU_CONSTANT inline static constexpr auto depos_prefactor_g = depos_prefactor;
#endif

template<int depos_order> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto get_depos_prefactor (int ix) {
#ifdef AMREX_USE_CUDA
    AMREX_IF_ON_DEVICE((return amrex::get<depos_order>(depos_prefactor_g)[ix];))
    AMREX_IF_ON_HOST((return amrex::get<depos_order>(depos_prefactor)[ix];))
#else
    return amrex::get<depos_order>(depos_prefactor)[ix];
#endif
}

/** \brief Compute a single shape factor and return the index of the cell where the particle writes.
 *
 * \tparam depos_order Order of the shape factor
 * \param[in] xmid exact position of the particle in index space
 * \param[in] ix index of the shape factor, must be 0 <= ix <= depos_order
 */
template<int depos_order> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
shape_factor_result shape_factor (amrex::Real xmid, int ix) noexcept {

    if constexpr (depos_order % 2 == 0) {
        xmid += amrex::Real(0.5);
    }

    const amrex::Real xfloor = std::floor(xmid);
    constexpr int floor_offset = - depos_order / 2;
    const int xbegin = static_cast<int>(xfloor) + floor_offset;

    const amrex::Real xint = xmid-xfloor;
    const auto s = get_depos_prefactor<depos_order>(ix);

    amrex::Real res = s[0];
    for (int i=1; i<=depos_order; ++i) {
        res = res * xint + s[i];
    }
    return {res, xbegin + ix};
}


#define WARPX_LOOP_UNROLL _Pragma("unroll")



/**
 * \brief Kernel for the direct current deposition for thread thread_num
 * \tparam depos_order deposition order
 * \param xp, yp, zp    The particle positions.
 * \param wq            The charge of the macroparticle
 * \param vx,vy,vz      The particle velocities
 * \param jx_arr,jy_arr,jz_arr Array4 of current density, either full array or tile.
 * \param jx_type,jy_type,jz_type The grid types along each direction, either NODE or CELL
 * \param relative_time Time at which to deposit J, relative to the time of the
 *                      current positions of the particles. When different than 0,
 *                      the particle position will be temporarily modified to match
 *                      the time of the deposition.
 * \param dinv          3D cell size inverse
 * \param xyzmin        The lower bounds of the domain
 * \param invvol        The inverse volume of a grid cell
 * \param lo            Index lower bounds of domain.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void doDepositionShapeNKernel2([[maybe_unused]] const amrex::ParticleReal xp,
                              [[maybe_unused]] const amrex::ParticleReal yp,
                              [[maybe_unused]] const amrex::ParticleReal zp,
                              const amrex::ParticleReal wq,
                              const amrex::ParticleReal vx,
                              const amrex::ParticleReal vy,
                              const amrex::ParticleReal vz,
                              amrex::Array4<amrex::Real> const& jx_arr,
                              amrex::Array4<amrex::Real> const& jy_arr,
                              amrex::Array4<amrex::Real> const& jz_arr,
                              amrex::IntVect const& jx_type,
                              amrex::IntVect const& jy_type,
                              amrex::IntVect const& jz_type,
                              const amrex::Real relative_time,
                              const amrex::XDim3 & dinv,
                              const amrex::XDim3 & xyzmin,
                              const amrex::Real invvol,
                              const amrex::Dim3 lo,
                              [[maybe_unused]] const int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

    constexpr int NODE = amrex::IndexType::NODE;
    constexpr int CELL = amrex::IndexType::CELL;

    const amrex::Real wqx = wq*invvol*vx;
    const amrex::Real wqy = wq*invvol*vy;
    const amrex::Real wqz = wq*invvol*vz;


    // --- Compute shape factors
    Compute_shape_factor< depos_order > const compute_shape_factor;

    // x direction
    // Get particle position after 1/2 push back in position
    // Keep these double to avoid bug in single precision

    /*


    // j_j[xyz] leftmost grid point in x that the particle touches for the centering of each current
    // sx_j[xyz] shape factor along x for the centering of each current
    // There are only two possible centerings, node or cell centered, so at most only two shape factor
    // arrays will be needed.
    // Keep these double to avoid bug in single precision
    const double xmid = ((xp - xyzmin.x) + relative_time*vx)*dinv.x;
    double sx_node[depos_order + 1] = {0.};
    double sx_cell[depos_order + 1] = {0.};
    int j_node = 0;
    int j_cell = 0;
    if (jx_type[0] == NODE || jy_type[0] == NODE || jz_type[0] == NODE) {
        j_node = compute_shape_factor(sx_node, xmid);
    }
    if (jx_type[0] == CELL || jy_type[0] == CELL || jz_type[0] == CELL) {
        j_cell = compute_shape_factor(sx_cell, xmid - 0.5);
    }

    amrex::Real sx_jx[depos_order + 1] = {0._rt};
    amrex::Real sx_jy[depos_order + 1] = {0._rt};
    amrex::Real sx_jz[depos_order + 1] = {0._rt};
    for (int ix=0; ix<=depos_order; ix++)
    {
        sx_jx[ix] = ((jx_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
        sx_jy[ix] = ((jy_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
        sx_jz[ix] = ((jz_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
    }

    int const j_jx = ((jx_type[0] == NODE) ? j_node : j_cell);
    int const j_jy = ((jy_type[0] == NODE) ? j_node : j_cell);
    int const j_jz = ((jz_type[0] == NODE) ? j_node : j_cell);



    // y direction
    // Keep these double to avoid bug in single precision
    const double ymid = ((yp - xyzmin.y) + relative_time*vy)*dinv.y;
    double sy_node[depos_order + 1] = {0.};
    double sy_cell[depos_order + 1] = {0.};
    int k_node = 0;
    int k_cell = 0;
    if (jx_type[1] == NODE || jy_type[1] == NODE || jz_type[1] == NODE) {
        k_node = compute_shape_factor(sy_node, ymid);
    }
    if (jx_type[1] == CELL || jy_type[1] == CELL || jz_type[1] == CELL) {
        k_cell = compute_shape_factor(sy_cell, ymid - 0.5);
    }
    amrex::Real sy_jx[depos_order + 1] = {0._rt};
    amrex::Real sy_jy[depos_order + 1] = {0._rt};
    amrex::Real sy_jz[depos_order + 1] = {0._rt};
    for (int iy=0; iy<=depos_order; iy++)
    {
        sy_jx[iy] = ((jx_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
        sy_jy[iy] = ((jy_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
        sy_jz[iy] = ((jz_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
    }
    int const k_jx = ((jx_type[1] == NODE) ? k_node : k_cell);
    int const k_jy = ((jy_type[1] == NODE) ? k_node : k_cell);
    int const k_jz = ((jz_type[1] == NODE) ? k_node : k_cell);



    // z direction
    // Keep these double to avoid bug in single precision
    constexpr int zdir = WARPX_ZINDEX;
    const double zmid = ((zp - xyzmin.z) + relative_time*vz)*dinv.z;
    double sz_node[depos_order + 1] = {0.};
    double sz_cell[depos_order + 1] = {0.};
    int l_node = 0;
    int l_cell = 0;
    if (jx_type[zdir] == NODE || jy_type[zdir] == NODE || jz_type[zdir] == NODE) {
        l_node = compute_shape_factor(sz_node, zmid);
    }
    if (jx_type[zdir] == CELL || jy_type[zdir] == CELL || jz_type[zdir] == CELL) {
        l_cell = compute_shape_factor(sz_cell, zmid - 0.5);
    }
    amrex::Real sz_jx[depos_order + 1] = {0._rt};
    amrex::Real sz_jy[depos_order + 1] = {0._rt};
    amrex::Real sz_jz[depos_order + 1] = {0._rt};
    for (int iz=0; iz<=depos_order; iz++)
    {
        sz_jx[iz] = ((jx_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
        sz_jy[iz] = ((jy_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
        sz_jz[iz] = ((jz_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
    }
    int const l_jx = ((jx_type[zdir] == NODE) ? l_node : l_cell);
    int const l_jy = ((jy_type[zdir] == NODE) ? l_node : l_cell);
    int const l_jz = ((jz_type[zdir] == NODE) ? l_node : l_cell);

    for (int ix=0; ix<=depos_order; ix++){
        for (int iy=0; iy<=depos_order; iy++){
            for (int iz=0; iz<=depos_order; iz++){
                amrex::Gpu::Atomic::AddNoRet(
                    &jx_arr(lo.x+j_jx+ix, lo.y+k_jx+iy, lo.z+l_jx+iz),
                    sx_jx[ix]*sy_jx[iy]*sz_jx[iz]*wqx);
                amrex::Gpu::Atomic::AddNoRet(
                    &jy_arr(lo.x+j_jy+ix, lo.y+k_jy+iy, lo.z+l_jy+iz),
                    sx_jy[ix]*sy_jy[iy]*sz_jy[iz]*wqy);
                amrex::Gpu::Atomic::AddNoRet(
                    &jz_arr(lo.x+j_jz+ix, lo.y+k_jz+iy, lo.z+l_jz+iz),
                    sx_jz[ix]*sy_jz[iy]*sz_jz[iz]*wqz);
            }
        }
    }

    */



    /*

    const double xmid = ((xp - xyzmin.x) + relative_time*vx)*dinv.x;
    const double ymid = ((yp - xyzmin.y) + relative_time*vy)*dinv.y;
    const double zmid = ((zp - xyzmin.z) + relative_time*vz)*dinv.z;


    amrex::Real sx_jx[depos_order + 1] = {0._rt};
    amrex::Real sx_jy[depos_order + 1] = {0._rt};
    amrex::Real sx_jz[depos_order + 1] = {0._rt};

    int const j_jx = ((jx_type[0] == NODE) ? compute_shape_factor(sx_jx, xmid) : compute_shape_factor(sx_jx, xmid - 0.5));
    int const j_jy = ((jy_type[0] == NODE) ? compute_shape_factor(sx_jy, xmid) : compute_shape_factor(sx_jy, xmid - 0.5));
    int const j_jz = ((jz_type[0] == NODE) ? compute_shape_factor(sx_jz, xmid) : compute_shape_factor(sx_jz, xmid - 0.5));


    amrex::Real sy_jx[depos_order + 1] = {0._rt};
    amrex::Real sy_jy[depos_order + 1] = {0._rt};
    amrex::Real sy_jz[depos_order + 1] = {0._rt};

    int const k_jx = ((jx_type[1] == NODE) ? compute_shape_factor(sy_jx, ymid) : compute_shape_factor(sy_jx, ymid - 0.5));
    int const k_jy = ((jy_type[1] == NODE) ? compute_shape_factor(sy_jy, ymid) : compute_shape_factor(sy_jy, ymid - 0.5));
    int const k_jz = ((jz_type[1] == NODE) ? compute_shape_factor(sy_jz, ymid) : compute_shape_factor(sy_jz, ymid - 0.5));


    amrex::Real sz_jx[depos_order + 1] = {0._rt};
    amrex::Real sz_jy[depos_order + 1] = {0._rt};
    amrex::Real sz_jz[depos_order + 1] = {0._rt};

    int const l_jx = ((jx_type[2] == NODE) ? compute_shape_factor(sz_jx, zmid) : compute_shape_factor(sz_jx, zmid - 0.5));
    int const l_jy = ((jy_type[2] == NODE) ? compute_shape_factor(sz_jy, zmid) : compute_shape_factor(sz_jy, zmid - 0.5));
    int const l_jz = ((jz_type[2] == NODE) ? compute_shape_factor(sz_jz, zmid) : compute_shape_factor(sz_jz, zmid - 0.5));


    for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<=depos_order; iy++){
            for (int ix=0; ix<=depos_order; ix++){
                amrex::Gpu::Atomic::AddNoRet(
                    &jx_arr(lo.x+j_jx+ix, lo.y+k_jx+iy, lo.z+l_jx+iz),
                    sx_jx[ix]*sy_jx[iy]*sz_jx[iz]*wqx);
                amrex::Gpu::Atomic::AddNoRet(
                    &jy_arr(lo.x+j_jy+ix, lo.y+k_jy+iy, lo.z+l_jy+iz),
                    sx_jy[ix]*sy_jy[iy]*sz_jy[iz]*wqy);
                amrex::Gpu::Atomic::AddNoRet(
                    &jz_arr(lo.x+j_jz+ix, lo.y+k_jz+iy, lo.z+l_jz+iz),
                    sx_jz[ix]*sy_jz[iy]*sz_jz[iz]*wqz);
            }
        }
    }

    */

    // const double xmid = ((xp - xyzmin.x) + relative_time*vx)*dinv.x;
    // const double ymid = ((yp - xyzmin.y) + relative_time*vy)*dinv.y;
    // const double zmid = ((zp - xyzmin.z) + relative_time*vz)*dinv.z;

    /*for (int ix=0; ix<=depos_order; ix++){
        for (int iy=0; iy<=depos_order; iy++){
            for (int iz=0; iz<=depos_order; iz++){

                auto [sx_jx, ix_jx] = shape_factor<depos_order>((jx_type[0] == NODE) ? xmid : xmid - 0.5_rt, ix);
                auto [sx_jy, ix_jy] = shape_factor<depos_order>((jy_type[0] == NODE) ? xmid : xmid - 0.5_rt, ix);
                auto [sx_jz, ix_jz] = shape_factor<depos_order>((jz_type[0] == NODE) ? xmid : xmid - 0.5_rt, ix);

                auto [sy_jx, iy_jx] = shape_factor<depos_order>((jx_type[1] == NODE) ? ymid : ymid - 0.5_rt, iy);
                auto [sy_jy, iy_jy] = shape_factor<depos_order>((jy_type[1] == NODE) ? ymid : ymid - 0.5_rt, iy);
                auto [sy_jz, iy_jz] = shape_factor<depos_order>((jz_type[1] == NODE) ? ymid : ymid - 0.5_rt, iy);

                auto [sz_jx, iz_jx] = shape_factor<depos_order>((jx_type[2] == NODE) ? zmid : zmid - 0.5_rt, iz);
                auto [sz_jy, iz_jy] = shape_factor<depos_order>((jy_type[2] == NODE) ? zmid : zmid - 0.5_rt, iz);
                auto [sz_jz, iz_jz] = shape_factor<depos_order>((jz_type[2] == NODE) ? zmid : zmid - 0.5_rt, iz);

                amrex::Gpu::Atomic::AddNoRet(&jx_arr(lo.x+ix_jx, lo.y+iy_jx, lo.z+iz_jx), sx_jx*sy_jx*sz_jx*wqx);
                amrex::Gpu::Atomic::AddNoRet(&jy_arr(lo.x+ix_jy, lo.y+iy_jy, lo.z+iz_jy), sx_jy*sy_jy*sz_jy*wqy);
                amrex::Gpu::Atomic::AddNoRet(&jz_arr(lo.x+ix_jz, lo.y+iy_jz, lo.z+iz_jz), sx_jz*sy_jz*sz_jz*wqz);
            }
        }
    }*/


    /*for (int ix=0; ix<=depos_order; ix++){
        for (int iy=0; iy<=depos_order; iy++){
           for (int iz=0; iz<=depos_order; iz++){
                auto [sx_jx, ix_jx] = shape_factor<depos_order>((jx_type[0] == NODE) ? xmid : xmid - 0.5_rt, ix);
                auto [sy_jx, iy_jx] = shape_factor<depos_order>((jx_type[1] == NODE) ? ymid : ymid - 0.5_rt, iy);
                auto [sz_jx, iz_jx] = shape_factor<depos_order>((jx_type[2] == NODE) ? zmid : zmid - 0.5_rt, iz);

                amrex::Gpu::Atomic::AddNoRet(&jx_arr(lo.x+ix_jx, lo.y+iy_jx, lo.z+iz_jx), sx_jx*sy_jx*sz_jx*wqx);
            }

            for (int iz=0; iz<=depos_order; iz++){
                auto [sx_jy, ix_jy] = shape_factor<depos_order>((jy_type[0] == NODE) ? xmid : xmid - 0.5_rt, ix);
                auto [sy_jy, iy_jy] = shape_factor<depos_order>((jy_type[1] == NODE) ? ymid : ymid - 0.5_rt, iy);
                auto [sz_jy, iz_jy] = shape_factor<depos_order>((jy_type[2] == NODE) ? zmid : zmid - 0.5_rt, iz);

                amrex::Gpu::Atomic::AddNoRet(&jy_arr(lo.x+ix_jy, lo.y+iy_jy, lo.z+iz_jy), sx_jy*sy_jy*sz_jy*wqy);
            }

            for (int iz=0; iz<=depos_order; iz++){
                auto [sx_jz, ix_jz] = shape_factor<depos_order>((jz_type[0] == NODE) ? xmid : xmid - 0.5_rt, ix);
                auto [sy_jz, iy_jz] = shape_factor<depos_order>((jz_type[1] == NODE) ? ymid : ymid - 0.5_rt, iy);
                auto [sz_jz, iz_jz] = shape_factor<depos_order>((jz_type[2] == NODE) ? zmid : zmid - 0.5_rt, iz);

                amrex::Gpu::Atomic::AddNoRet(&jz_arr(lo.x+ix_jz, lo.y+iy_jz, lo.z+iz_jz), sx_jz*sy_jz*sz_jz*wqz);
            }
        }
    }*/

    const double xmid = ((xp - xyzmin.x) + relative_time*vx)*dinv.x;
    const double ymid = ((yp - xyzmin.y) + relative_time*vy)*dinv.y;
    const double zmid = ((zp - xyzmin.z) + relative_time*vz)*dinv.z;

    _Pragma("unroll 1")
    for (int ix=0; ix<=depos_order; ix++){

        auto [sx_jx, ix_jx] = shape_factor<depos_order>((jx_type[0] == NODE) ? xmid : xmid - 0.5_rt, ix);
        auto [sx_jy, ix_jy] = shape_factor<depos_order>((jy_type[0] == NODE) ? xmid : xmid - 0.5_rt, ix);
        auto [sx_jz, ix_jz] = shape_factor<depos_order>((jz_type[0] == NODE) ? xmid : xmid - 0.5_rt, ix);

        _Pragma("unroll 1")
        for (int iy=0; iy<=depos_order; iy++){

            auto [sy_jx, iy_jx] = shape_factor<depos_order>((jx_type[1] == NODE) ? ymid : ymid - 0.5_rt, iy);
            auto [sy_jy, iy_jy] = shape_factor<depos_order>((jy_type[1] == NODE) ? ymid : ymid - 0.5_rt, iy);
            auto [sy_jz, iy_jz] = shape_factor<depos_order>((jz_type[1] == NODE) ? ymid : ymid - 0.5_rt, iy);

            _Pragma("unroll")
            for (int iz=0; iz<=depos_order; iz++){

                auto [sz_jx, iz_jx] = shape_factor<depos_order>((jx_type[2] == NODE) ? zmid : zmid - 0.5_rt, iz);
                auto [sz_jy, iz_jy] = shape_factor<depos_order>((jy_type[2] == NODE) ? zmid : zmid - 0.5_rt, iz);
                auto [sz_jz, iz_jz] = shape_factor<depos_order>((jz_type[2] == NODE) ? zmid : zmid - 0.5_rt, iz);

                amrex::Gpu::Atomic::AddNoRet(&jx_arr(lo.x+ix_jx, lo.y+iy_jx, lo.z+iz_jx), wqx*sx_jx*sy_jx*sz_jx);
                amrex::Gpu::Atomic::AddNoRet(&jy_arr(lo.x+ix_jy, lo.y+iy_jy, lo.z+iz_jy), wqy*sx_jy*sy_jy*sz_jy);
                amrex::Gpu::Atomic::AddNoRet(&jz_arr(lo.x+ix_jz, lo.y+iy_jz, lo.z+iz_jz), wqz*sx_jz*sy_jz*sz_jz);
            }
        }
    }

    /*

    const double xmid = ((xp - xyzmin.x) + relative_time*vx)*dinv.x;
    const double ymid = ((yp - xyzmin.y) + relative_time*vy)*dinv.y;
    const double zmid = ((zp - xyzmin.z) + relative_time*vz)*dinv.z;


    amrex::Real sx_jx[depos_order + 1] = {0._rt};
    amrex::Real sx_jy[depos_order + 1] = {0._rt};
    amrex::Real sx_jz[depos_order + 1] = {0._rt};

    int const j_jx = ((jx_type[0] == NODE) ? compute_shape_factor(sx_jx, xmid) : compute_shape_factor(sx_jx, xmid - 0.5));
    int const j_jy = ((jy_type[0] == NODE) ? compute_shape_factor(sx_jy, xmid) : compute_shape_factor(sx_jy, xmid - 0.5));
    int const j_jz = ((jz_type[0] == NODE) ? compute_shape_factor(sx_jz, xmid) : compute_shape_factor(sx_jz, xmid - 0.5));


    amrex::Real sy_jx[depos_order + 1] = {0._rt};
    amrex::Real sy_jy[depos_order + 1] = {0._rt};
    amrex::Real sy_jz[depos_order + 1] = {0._rt};

    int const k_jx = ((jx_type[1] == NODE) ? compute_shape_factor(sy_jx, ymid) : compute_shape_factor(sy_jx, ymid - 0.5));
    int const k_jy = ((jy_type[1] == NODE) ? compute_shape_factor(sy_jy, ymid) : compute_shape_factor(sy_jy, ymid - 0.5));
    int const k_jz = ((jz_type[1] == NODE) ? compute_shape_factor(sy_jz, ymid) : compute_shape_factor(sy_jz, ymid - 0.5));


    amrex::Real sz_jx[depos_order + 1] = {0._rt};
    amrex::Real sz_jy[depos_order + 1] = {0._rt};
    amrex::Real sz_jz[depos_order + 1] = {0._rt};

    int const l_jx = ((jx_type[2] == NODE) ? compute_shape_factor(sz_jx, zmid) : compute_shape_factor(sz_jx, zmid - 0.5));
    int const l_jy = ((jy_type[2] == NODE) ? compute_shape_factor(sz_jy, zmid) : compute_shape_factor(sz_jy, zmid - 0.5));
    int const l_jz = ((jz_type[2] == NODE) ? compute_shape_factor(sz_jz, zmid) : compute_shape_factor(sz_jz, zmid - 0.5));


    for (int ix=0; ix<=depos_order; ix++){
        for (int iy=0; iy<=depos_order; iy++){
            for (int iz=0; iz<=depos_order; iz++){
                amrex::Gpu::Atomic::AddNoRet(
                    &jx_arr(lo.x+j_jx+ix, lo.y+k_jx+iy, lo.z+l_jx+iz),
                    sx_jx[ix]*sy_jx[iy]*sz_jx[iz]*wqx);
                amrex::Gpu::Atomic::AddNoRet(
                    &jy_arr(lo.x+j_jy+ix, lo.y+k_jy+iy, lo.z+l_jy+iz),
                    sx_jy[ix]*sy_jy[iy]*sz_jy[iz]*wqy);
                amrex::Gpu::Atomic::AddNoRet(
                    &jz_arr(lo.x+j_jz+ix, lo.y+k_jz+iy, lo.z+l_jz+iz),
                    sx_jz[ix]*sy_jz[iy]*sz_jz[iz]*wqz);
            }
        }
    }*/
}

/**
 * \brief Current Deposition for thread thread_num
 * \tparam depos_order deposition order
 * \param GetPosition  A functor for returning the particle position.
 * \param wp           Pointer to array of particle weights.
 * \param uxp,uyp,uzp  Pointer to arrays of particle momentum.
 * \param ion_lev      Pointer to array of particle ionization level. This is
                         required to have the charge of each macroparticle
                         since q is a scalar. For non-ionizable species,
                         ion_lev is a null pointer.
 * \param jx_fab,jy_fab,jz_fab FArrayBox of current density, either full array or tile.
 * \param np_to_deposit Number of particles for which current is deposited.
 * \param relative_time Time at which to deposit J, relative to the time of the
 *                      current positions of the particles. When different than 0,
 *                      the particle position will be temporarily modified to match
 *                      the time of the deposition.
 * \param dinv         3D cell size inverse
 * \param xyzmin       Physical lower bounds of domain.
 * \param lo           Index lower bounds of domain.
 * \param q            species charge.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
void doDepositionShapeN (const GetParticlePosition<PIdx>& GetPosition,
                         const amrex::ParticleReal * const wp,
                         const amrex::ParticleReal * const uxp,
                         const amrex::ParticleReal * const uyp,
                         const amrex::ParticleReal * const uzp,
                         const int* ion_lev,
                         amrex::FArrayBox& jx_fab,
                         amrex::FArrayBox& jy_fab,
                         amrex::FArrayBox& jz_fab,
                         long np_to_deposit,
                         amrex::Real relative_time,
                         const amrex::XDim3 & dinv,
                         const amrex::XDim3 & xyzmin,
                         amrex::Dim3 lo,
                         amrex::Real q,
                         [[maybe_unused]]int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    const amrex::Real clightsq = 1.0_rt/PhysConst::c/PhysConst::c;

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();
    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();

    // std::cout << "TYPE: " << jx_type << " " << jy_type << " " << jz_type << std::endl;

    // Loop over particles and deposit into jx_fab, jy_fab and jz_fab
    amrex::ParallelFor(
        np_to_deposit,
        [=] AMREX_GPU_DEVICE (long ip) {
            amrex::ParticleReal xp, yp, zp;
            GetPosition(ip, xp, yp, zp);

            // --- Get particle quantities
            const amrex::Real gaminv = amrex::Math::rsqrt(1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                                        + uyp[ip]*uyp[ip]*clightsq
                                                        + uzp[ip]*uzp[ip]*clightsq);
            const amrex::Real vx  = uxp[ip]*gaminv;
            const amrex::Real vy  = uyp[ip]*gaminv;
            const amrex::Real vz  = uzp[ip]*gaminv;

            amrex::Real wq  = q*wp[ip];
            if (do_ionization){
                wq *= ion_lev[ip];
            }

            doDepositionShapeNKernel2<depos_order>(xp, yp, zp, wq, vx, vy, vz, jx_arr, jy_arr, jz_arr,
                                                  jx_type, jy_type, jz_type,
                                                  relative_time, dinv, xyzmin,
                                                  invvol, lo, n_rz_azimuthal_modes);

        }
    );
}

template void doDepositionShapeN<1>(const GetParticlePosition<PIdx>& GetPosition,
                         const amrex::ParticleReal * const wp,
                         const amrex::ParticleReal * const uxp,
                         const amrex::ParticleReal * const uyp,
                         const amrex::ParticleReal * const uzp,
                         const int* ion_lev,
                         amrex::FArrayBox& jx_fab,
                         amrex::FArrayBox& jy_fab,
                         amrex::FArrayBox& jz_fab,
                         long np_to_deposit,
                         amrex::Real relative_time,
                         const amrex::XDim3 & dinv,
                         const amrex::XDim3 & xyzmin,
                         amrex::Dim3 lo,
                         amrex::Real q,
                         [[maybe_unused]]int n_rz_azimuthal_modes);

template void doDepositionShapeN<2>(const GetParticlePosition<PIdx>& GetPosition,
                         const amrex::ParticleReal * const wp,
                         const amrex::ParticleReal * const uxp,
                         const amrex::ParticleReal * const uyp,
                         const amrex::ParticleReal * const uzp,
                         const int* ion_lev,
                         amrex::FArrayBox& jx_fab,
                         amrex::FArrayBox& jy_fab,
                         amrex::FArrayBox& jz_fab,
                         long np_to_deposit,
                         amrex::Real relative_time,
                         const amrex::XDim3 & dinv,
                         const amrex::XDim3 & xyzmin,
                         amrex::Dim3 lo,
                         amrex::Real q,
                         [[maybe_unused]]int n_rz_azimuthal_modes);

template void doDepositionShapeN<3>(const GetParticlePosition<PIdx>& GetPosition,
                         const amrex::ParticleReal * const wp,
                         const amrex::ParticleReal * const uxp,
                         const amrex::ParticleReal * const uyp,
                         const amrex::ParticleReal * const uzp,
                         const int* ion_lev,
                         amrex::FArrayBox& jx_fab,
                         amrex::FArrayBox& jy_fab,
                         amrex::FArrayBox& jz_fab,
                         long np_to_deposit,
                         amrex::Real relative_time,
                         const amrex::XDim3 & dinv,
                         const amrex::XDim3 & xyzmin,
                         amrex::Dim3 lo,
                         amrex::Real q,
                         [[maybe_unused]]int n_rz_azimuthal_modes);

template void doDepositionShapeN<4>(const GetParticlePosition<PIdx>& GetPosition,
                         const amrex::ParticleReal * const wp,
                         const amrex::ParticleReal * const uxp,
                         const amrex::ParticleReal * const uyp,
                         const amrex::ParticleReal * const uzp,
                         const int* ion_lev,
                         amrex::FArrayBox& jx_fab,
                         amrex::FArrayBox& jy_fab,
                         amrex::FArrayBox& jz_fab,
                         long np_to_deposit,
                         amrex::Real relative_time,
                         const amrex::XDim3 & dinv,
                         const amrex::XDim3 & xyzmin,
                         amrex::Dim3 lo,
                         amrex::Real q,
                         [[maybe_unused]]int n_rz_azimuthal_modes);