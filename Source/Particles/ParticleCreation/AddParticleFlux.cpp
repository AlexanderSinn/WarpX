/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Particles/PhysicalParticleContainer.H"

#include "AddPlasmaUtilities.H"
#include "DefaultInitialization.H"
#include "Initialization/InjectorDensity.H"
#include "Initialization/InjectorMomentum.H"
#include "Initialization/InjectorPosition.H"
#ifdef WARPX_QED
#   include "Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.H"
#   include "Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.H"
#endif
#include "Particles/Pusher/UpdatePosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/ParticleUtils.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "Utils/WarpXConst.H"
#include "EmbeddedBoundary/Enabled.H"
#ifdef AMREX_USE_EB
#   include "EmbeddedBoundary/ParticleBoundaryProcess.H"
#   include "EmbeddedBoundary/ParticleScraper.H"
#endif
#include "WarpX.H"

#include <ablastr/warn_manager/WarnManager.H>
#include <ablastr/utils/Communication.H>

#include <AMReX.H>
#include <AMReX_Algorithm.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_Dim3.H>
#include <AMReX_Extension.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_INT.H>
#include <AMReX_IntVect.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParGDB.H>
#include <AMReX_ParIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particle.H>
#include <AMReX_ParticleContainerBase.H>
#include <AMReX_AmrParticles.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_Random.H>
#include <AMReX_SPACE.H>
#include <AMReX_StructOfArrays.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>
#include <AMReX_Parser.H>

#ifdef AMREX_USE_OMP
#   include <omp.h>
#endif

#ifdef WARPX_USE_OPENPMD
#   include <openPMD/openPMD.hpp>
#endif

#include <any>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <sstream>

using namespace amrex;

namespace
{
    using ParticleType = WarpXParticleContainer::ParticleType;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    XDim3 getCellCoords (const amrex::GpuArray<Real, AMREX_SPACEDIM>& lo_corner,
                         const amrex::GpuArray<Real, AMREX_SPACEDIM>& dx,
                         const XDim3& r, const amrex::IntVect& iv) noexcept
    {
        XDim3 pos;
#if defined(WARPX_DIM_3D)
        pos.x = lo_corner[0] + (iv[0]+r.x)*dx[0];
        pos.y = lo_corner[1] + (iv[1]+r.y)*dx[1];
        pos.z = lo_corner[2] + (iv[2]+r.z)*dx[2];
#elif defined(WARPX_DIM_XZ)
        pos.x = lo_corner[0] + (iv[0]+r.x)*dx[0];
        pos.y = 0.0_rt;
        pos.z = lo_corner[1] + (iv[1]+r.y)*dx[1];
#elif defined(WARPX_DIM_RZ)
        // Note that for RZ, r.y will be theta
        pos.x = lo_corner[0] + (iv[0]+r.x)*dx[0];
        pos.y = 0.0_rt;
        pos.z = lo_corner[1] + (iv[1]+r.z)*dx[1];
#elif defined(WARPX_DIM_1D_Z)
        pos.x = 0.0_rt;
        pos.y = 0.0_rt;
        pos.z = lo_corner[0] + (iv[0]+r.x)*dx[0];
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        pos.x = lo_corner[0] + (iv[0]+r.x)*dx[0];
        pos.y = 0.0_rt;
        pos.z = 0.0_rt;
#endif
        return pos;
    }
}

void
PhysicalParticleContainer::AddPlasmaFlux (PlasmaInjector const& plasma_injector, amrex::Real dt)
{
    WARPX_PROFILE("PhysicalParticleContainer::AddPlasmaFlux()");

    const Geometry& geom = Geom(0);
    const amrex::RealBox& part_realbox = geom.ProbDomain();

    const amrex::Real num_ppc_real = plasma_injector.num_particles_per_cell_real;
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    const amrex::Real rmax = std::min(plasma_injector.xmax, geom.ProbDomain().hi(0));
    const amrex::Real rmin = std::max(plasma_injector.xmin, geom.ProbDomain().lo(0));
#endif

    const auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();

#ifdef AMREX_USE_EB
    bool const inject_from_eb = plasma_injector.m_inject_from_eb; // whether to inject from EB or from a plane
    // Extract data structures for embedded boundaries
    amrex::EBFArrayBoxFactory const* eb_factory = nullptr;
    amrex::FabArray<amrex::EBCellFlagFab> const* eb_flag = nullptr;
    if (inject_from_eb) {
        eb_factory = &(WarpX::GetInstance().fieldEBFactory(0));
        eb_flag = &(eb_factory->getMultiEBCellFlagFab());
    }
#endif

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(0);

    // Create temporary particle container to which particles will be added;
    // we will then call Redistribute on this new container and finally
    // add the new particles to the original container.
    PhysicalParticleContainer tmp_pc(&WarpX::GetInstance());
    for (int ic = 0; ic < NumRuntimeRealComps(); ++ic) { tmp_pc.AddRealComp(GetRealSoANames()[ic + NArrayReal], false); }
    for (int ic = 0; ic < NumRuntimeIntComps(); ++ic) { tmp_pc.AddIntComp(GetIntSoANames()[ic + NArrayInt], false); }
    tmp_pc.defineAllParticleTiles();

    amrex::Box fine_injection_box;
    amrex::IntVect rrfac(AMREX_D_DECL(1,1,1));
    const bool refine_injection = findRefinedInjectionBox(fine_injection_box, rrfac);

    InjectorPosition* flux_pos = plasma_injector.getInjectorFluxPosition();
    InjectorFlux*  inj_flux = plasma_injector.getInjectorFlux();
    InjectorMomentum* inj_mom = plasma_injector.getInjectorMomentumDevice();
    constexpr int level_zero = 0;
    const amrex::Real t = WarpX::GetInstance().gett_new(level_zero);

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
    const int nmodes = WarpX::n_rz_azimuthal_modes;
    const bool rz_random_theta = m_rz_random_theta;
#endif
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    const amrex::Real radial_numpercell_power = plasma_injector.radial_numpercell_power;
#endif

    auto n_user_int_attribs = static_cast<int>(m_user_int_attribs.size());
    auto n_user_real_attribs = static_cast<int>(m_user_real_attribs.size());
    const PlasmaParserWrapper plasma_parser_wrapper (m_user_int_attribs.size(),
                                                     m_user_real_attribs.size(),
                                                     m_user_int_attrib_parser,
                                                     m_user_real_attrib_parser);

    MFItInfo info;
    if (do_tiling && amrex::Gpu::notInLaunchRegion()) {
        info.EnableTiling(tile_size);
    }
#ifdef AMREX_USE_OMP
    info.SetDynamic(true);
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi = MakeMFIter(0, info); mfi.isValid(); ++mfi)
    {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        const amrex::Box& tile_box = mfi.tilebox();
        const amrex::RealBox tile_realbox = WarpX::getRealBox(tile_box, 0);

        // Find the cells of part_realbox that overlap with tile_realbox
        // If there is no overlap, just go to the next tile in the loop
        amrex::RealBox overlap_realbox;
        amrex::Box overlap_box;
        amrex::IntVect shifted;
#ifdef AMREX_USE_EB
        if (inject_from_eb) {
            // Injection from EB
            const amrex::FabType fab_type = (*eb_flag)[mfi].getType(tile_box);
            if (fab_type == amrex::FabType::regular) { continue; } // Go to the next tile
            if (fab_type == amrex::FabType::covered) { continue; } // Go to the next tile
            overlap_box = tile_box;
            overlap_realbox = part_realbox;
        } else
#endif
        {
            // Injection from a plane
            const bool no_overlap = find_overlap_flux(tile_realbox, part_realbox, dx, problo, plasma_injector, overlap_realbox, overlap_box, shifted);
            if (no_overlap) { continue; } // Go to the next tile
        }

        const int grid_id = mfi.index();
        const int tile_id = mfi.LocalTileIndex();

        const amrex::GpuArray<Real,AMREX_SPACEDIM> overlap_corner
            {AMREX_D_DECL(overlap_realbox.lo(0),
                          overlap_realbox.lo(1),
                          overlap_realbox.lo(2))};

        // count the number of particles that each cell in overlap_box could add
        amrex::Gpu::DeviceVector<int> counts(overlap_box.numPts(), 0);
        amrex::Gpu::DeviceVector<int> offset(overlap_box.numPts());
        auto *pcounts = counts.data();
        const int flux_normal_axis = plasma_injector.flux_normal_axis;
        amrex::Box fine_overlap_box; // default Box is NOT ok().
        if (refine_injection) {
            fine_overlap_box = overlap_box & amrex::shift(fine_injection_box, -shifted);
        }

#ifdef AMREX_USE_EB
        auto eb_flag_arr = eb_flag ? eb_flag->const_array(mfi) : Array4<EBCellFlag const>{};
        auto eb_data = eb_factory ? eb_factory->getEBData(mfi) : EBData{};
#endif

        amrex::ParallelForRNG(overlap_box, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
            amrex::ignore_unused(j,k);

            // Determine the number of macroparticles to inject in this cell (num_ppc_int)
#ifdef AMREX_USE_EB
            amrex::Real num_ppc_real_in_this_cell = num_ppc_real; // user input: number of macroparticles per cell
            if (inject_from_eb) {
                // Injection from EB
                // Skip cells that are not partially covered by the EB
                if (eb_flag_arr(i,j,k).isRegular() || eb_flag_arr(i,j,k).isCovered()) { return; }
                // Scale by the (normalized) area of the EB surface in this cell
                num_ppc_real_in_this_cell *= eb_data.get<amrex::EBData_t::bndryarea>(i,j,k);
            }
#else
            amrex::Real const num_ppc_real_in_this_cell = num_ppc_real; // user input: number of macroparticles per cell
#endif
            // Skip cells that do not overlap with the bounds specified by the user (xmin/xmax, ymin/ymax, zmin/zmax)
            auto lo = getCellCoords(overlap_corner, dx, {0._rt, 0._rt, 0._rt}, iv);
            auto hi = getCellCoords(overlap_corner, dx, {1._rt, 1._rt, 1._rt}, iv);
            if (!flux_pos->overlapsWith(lo, hi)) { return; }

            auto index = overlap_box.index(iv);
            // Take into account refined injection region
            int r = 1;
            if (fine_overlap_box.ok() && fine_overlap_box.contains(iv)) {
                r = compute_area_weights(rrfac, flux_normal_axis);
            }
            const int num_ppc_int = static_cast<int>(num_ppc_real_in_this_cell*r + amrex::Random(engine));
            pcounts[index] = num_ppc_int;

            amrex::ignore_unused(j,k);
        });

        // Max number of new particles. All of them are created,
        // and invalid ones are then discarded
        const amrex::Long max_new_particles = amrex::Scan::ExclusiveSum(counts.size(), counts.data(), offset.data());

        // Update NextID to include particles created in this function
        amrex::Long pid;
#ifdef AMREX_USE_OMP
#pragma omp critical (add_plasma_nextid)
#endif
        {
            pid = ParticleType::NextID();
            ParticleType::NextID(pid+max_new_particles);
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            pid + max_new_particles < LongParticleIds::LastParticleID,
            "overflow on particle id numbers");

        const int cpuid = ParallelDescriptor::MyProc();

        auto& particle_tile = tmp_pc.DefineAndReturnParticleTile(0, grid_id, tile_id);

        auto const old_size = static_cast<amrex::Long>(particle_tile.size());
        auto const new_size = old_size + max_new_particles;
        particle_tile.resize(new_size);

        auto& soa = particle_tile.GetStructOfArrays();
        amrex::GpuArray<ParticleReal*,PIdx::nattribs> pa;
        for (int ia = 0; ia < PIdx::nattribs; ++ia) {
            pa[ia] = soa.GetRealData(ia).data() + old_size;
        }
        uint64_t * AMREX_RESTRICT pa_idcpu = soa.GetIdCPUData().data() + old_size;

        PlasmaParserHelper plasma_parser_helper(soa, old_size, m_user_int_attribs, m_user_real_attribs, plasma_parser_wrapper);
        int** pa_user_int_data = plasma_parser_helper.getUserIntDataPtrs();
        amrex::ParticleReal** pa_user_real_data = plasma_parser_helper.getUserRealDataPtrs();
        amrex::ParserExecutor<7> const* user_int_parserexec_data = plasma_parser_helper.getUserIntParserExecData();
        amrex::ParserExecutor<7> const* user_real_parserexec_data = plasma_parser_helper.getUserRealParserExecData();

        int* p_ion_level = nullptr;
        if (do_field_ionization) {
            p_ion_level = soa.GetIntData("ionizationLevel").data() + old_size;
        }

#ifdef WARPX_QED
        const QEDHelper qed_helper(soa, old_size,
                                   has_quantum_sync(), has_breit_wheeler(),
                                   m_shr_p_qs_engine, m_shr_p_bw_engine);
#endif

        const bool loc_do_field_ionization = do_field_ionization;
        const int loc_ionization_initial_level = ionization_initial_level;
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        int const loc_flux_normal_axis = plasma_injector.flux_normal_axis;
#endif

        // Loop over all new particles and inject them (creates too many
        // particles, in particular does not consider xmin, xmax etc.).
        // The invalid ones are given negative ID and are deleted during the
        // next redistribute.
        auto *const poffset = offset.data();
        amrex::ParallelForRNG(overlap_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            const amrex::IntVect iv = amrex::IntVect(AMREX_D_DECL(i, j, k));
            amrex::ignore_unused(j,k);
            const auto index = overlap_box.index(iv);

            amrex::Real scale_fac;
#ifdef AMREX_USE_EB
            if (inject_from_eb) {
                scale_fac = compute_scale_fac_area_eb(dx, num_ppc_real,
                                                      AMREX_D_DECL(eb_data.get<amrex::EBData_t::bndrynorm>(i,j,k,0),
                                                                   eb_data.get<amrex::EBData_t::bndrynorm>(i,j,k,1),
                                                                   eb_data.get<amrex::EBData_t::bndrynorm>(i,j,k,2)));
            } else
#endif
            {
                scale_fac = compute_scale_fac_area_plane(dx, num_ppc_real, flux_normal_axis);
            }

            if (fine_overlap_box.ok() && fine_overlap_box.contains(iv)) {
                scale_fac /= compute_area_weights(rrfac, flux_normal_axis);
            }

            for (int i_part = 0; i_part < pcounts[index]; ++i_part)
            {
                const long ip = poffset[index] + i_part;
                pa_idcpu[ip] = amrex::SetParticleIDandCPU(pid+ip, cpuid);

                // Determine the position of the particle within the cell
                XDim3 pos;
                XDim3 r;
#ifdef AMREX_USE_EB
                if (inject_from_eb) {
                    auto const& pt = eb_data.randomPointOnEB(i,j,k,engine);
#if defined(WARPX_DIM_3D)
                    pos.x = overlap_corner[0] + (iv[0] + 0.5_rt + pt[0])*dx[0];
                    pos.y = overlap_corner[1] + (iv[1] + 0.5_rt + pt[1])*dx[1];
                    pos.z = overlap_corner[2] + (iv[2] + 0.5_rt + pt[2])*dx[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    pos.x = overlap_corner[0] + (iv[0] + 0.5_rt + pt[0])*dx[0];
                    pos.y = 0.0_rt;
                    pos.z = overlap_corner[1] + (iv[1] + 0.5_rt + pt[1])*dx[1];
#endif
                } else
#endif
                {
                    // Injection from a plane
                    // This assumes the flux_pos is of type InjectorPositionRandomPlane
                    r = (fine_overlap_box.ok() && fine_overlap_box.contains(iv)) ?
                        // In the refined injection region: use refinement ratio `rrfac`
                        flux_pos->getPositionUnitBox(i_part, rrfac, engine) :
                        // Otherwise: use 1 as the refinement ratio
                        flux_pos->getPositionUnitBox(i_part, amrex::IntVect::TheUnitVector(), engine);
                    pos = getCellCoords(overlap_corner, dx, r, iv);
                }
                auto ppos = PDim3(pos);

                // inj_mom would typically be InjectorMomentumGaussianFlux
                XDim3 u;
                u = inj_mom->getMomentum(pos.x, pos.y, pos.z, engine);
                auto pu = PDim3(u);

                pu.x *= PhysConst::c;
                pu.y *= PhysConst::c;
                pu.z *= PhysConst::c;

                // The containsInclusive is used to allow the case of the flux surface
                // being on the boundary of the domain. After the UpdatePosition below,
                // the particles will be within the domain.
#if defined(WARPX_DIM_3D)
                if (!ParticleUtils::containsInclusive(tile_realbox, XDim3{ppos.x,ppos.y,ppos.z})) {
                    pa_idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                    continue;
                }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                amrex::ignore_unused(k);
                if (!ParticleUtils::containsInclusive(tile_realbox, XDim3{ppos.x,ppos.z,0.0_prt})) {
                    pa_idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                    continue;
                }
#elif defined(WARPX_DIM_1D_Z)
                amrex::ignore_unused(j,k);
                if (!ParticleUtils::containsInclusive(tile_realbox, XDim3{ppos.z,0.0_prt,0.0_prt})) {
                    pa_idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                    continue;
                }
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                amrex::ignore_unused(j,k);
                if (!ParticleUtils::containsInclusive(tile_realbox, XDim3{ppos.x,0.0_prt,0.0_prt})) {
                    pa_idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                    continue;
                }
#endif
                // Lab-frame simulation
                // If the particle's initial position is not within or on the species's
                // xmin, xmax, ymin, ymax, zmin, zmax, go to the next generated particle.
                if (!flux_pos->insideBoundsInclusive(ppos.x, ppos.y, ppos.z)) {
                    pa_idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                    continue;
                }

#ifdef AMREX_USE_EB
                if (inject_from_eb) {
                    // Injection from EB: rotate momentum according to the normal of the EB surface
                    // (The above code initialized the momentum by assuming that z is the direction
                    // normal to the EB surface. Thus we need to rotate from z to the normal.)
                    rotate_momentum_eb(pu, AMREX_D_DECL(eb_data.get<amrex::EBData_t::bndrynorm>(i,j,k,0),
                                                        eb_data.get<amrex::EBData_t::bndrynorm>(i,j,k,1),
                                                        eb_data.get<amrex::EBData_t::bndrynorm>(i,j,k,2)));
                }
#endif

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                // Adjust the particle radius to produce the correct distribution.
                // Note that this may shift particles outside of the current tile,
                // but this is Ok since particles will be redistributed afterwards.
                // The containsInclusive check above ensures
                // that the "logical" space is uniformly filled.
                amrex::Real const xu = (ppos.x - rmin)/(rmax - rmin);
                amrex::Real const rc = std::pow(rmax, 1._rt + radial_numpercell_power)
                                     - std::pow(rmin, 1._rt + radial_numpercell_power);
                amrex::Real const rminp = std::pow(rmin, 1._rt + radial_numpercell_power);
                amrex::Real const radial_position = std::pow(xu*rc + rminp, 1._rt/(1._rt + radial_numpercell_power));

                // Conversion from cylindrical to Cartesian coordinates
                // Replace the x and y, setting an angle theta.
                // These x and y are used to get the momentum and flux
                // With only 1 mode, the angle doesn't matter so
                // choose it randomly.
                const amrex::Real theta = (nmodes == 1 && rz_random_theta)?
#if defined(WARPX_DIM_RZ)
                    // This should be updated to be the same as below, since theta
                    // should range from -pi to +pi. This should be a separate PR
                    // since it will break RZ CI tests.
                    (2._rt*MathConst::pi*amrex::Random(engine)):
#elif defined(WARPX_DIM_RCYLINDER)
                    (MathConst::pi*(2._rt*amrex::Random(engine) - 1._rt)):
#endif
                    (2._prt*MathConst::pi*r.y);
                amrex::Real const cos_theta = std::cos(theta);
                amrex::Real const sin_theta = std::sin(theta);
                // Rotate the position
                ppos.x = radial_position*cos_theta;
                ppos.y = radial_position*sin_theta;
                if ((loc_flux_normal_axis != 2)
#ifdef AMREX_USE_EB
                    || (inject_from_eb)
#endif
                    ) {
                    // Rotate the momentum
                    // This because, when the flux direction is e.g. "r"
                    // the `inj_mom` objects generates a v*Gaussian distribution
                    // along the Cartesian "x" direction by default. This
                    // needs to be rotated along "r".
                    const amrex::Real ur = pu.x;
                    const amrex::Real ut = pu.y;
                    pu.x = cos_theta*ur - sin_theta*ut;
                    pu.y = sin_theta*ur + cos_theta*ut;
                }
#elif defined(WARPX_DIM_RSPHERE)
                // Adjust the particle radius to produce the correct distribution.
                // Note that this may shift particles outside of the current tile,
                // but this is Ok since particles will be redistributed afterwards.
                // The containsInclusive check above ensures
                // that the "logical" space is uniformly filled.
                amrex::Real const xu = (ppos.x - rmin)/(rmax - rmin);
                amrex::Real const rc = std::pow(rmax, 1._rt + radial_numpercell_power)
                                     - std::pow(rmin, 1._rt + radial_numpercell_power);
                amrex::Real const rminp = std::pow(rmin, 1._rt + radial_numpercell_power);
                amrex::Real const radial_position = std::pow(xu*rc + rminp, 1._rt/(1._rt + radial_numpercell_power));

                // Replace the x, y, and z, setting angles theta and phi.
                // These x, y, and z are used to get the momentum and flux
                amrex::Real const theta = MathConst::pi*(2._rt*amrex::Random(engine) - 1._rt);
                amrex::Real const sin_phi = 2._rt*amrex::Random(engine) - 1._rt;
                amrex::Real const cos_phi = std::sqrt(1._rt - sin_phi*sin_phi);
                amrex::Real const phi = std::atan2(sin_phi, cos_phi);
                amrex::Real const cos_theta = std::cos(theta);
                amrex::Real const sin_theta = std::sin(theta);
                pos.x = radial_position*cos_phi*std::cos(theta);
                pos.y = radial_position*cos_phi*std::sin(theta);
                pos.z = radial_position*sin_phi;
                // Rotate the momentum
                // This because, when the flux direction is e.g. "r"
                // the `inj_mom` objects generates a v*Gaussian distribution
                // along the Cartesian "x" direction by default. This
                // needs to be rotated along "r".
                amrex::Real const ur = pu.x;
                amrex::Real const ut = pu.y;
                amrex::Real const up = pu.z;
                pu.x = cos_phi*cos_theta*ur - sin_theta*ut - sin_phi*cos_theta*up;
                pu.y = cos_phi*sin_theta*ur + cos_theta*ut - sin_phi*sin_theta*up;
                pu.z = sin_phi*ur + cos_phi*up;
#endif
                const amrex::Real flux = inj_flux->getFlux(ppos.x, ppos.y, ppos.z, t);
                // Remove particle if flux is negative or 0
                if (flux <= 0) {
                    pa_idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                    continue;
                }

                if (loc_do_field_ionization) {
                    p_ion_level[ip] = loc_ionization_initial_level;
                }

#ifdef WARPX_QED
                if(qed_helper.has_quantum_sync){
                    qed_helper.p_optical_depth_QSR[ip] = qed_helper.quantum_sync_get_opt(engine);
                }

                if(qed_helper.has_breit_wheeler){
                    qed_helper.p_optical_depth_BW[ip] = qed_helper.breit_wheeler_get_opt(engine);
                }
#endif

                // Initialize user-defined integers with user-defined parser
                for (int ia = 0; ia < n_user_int_attribs; ++ia) {
                    pa_user_int_data[ia][ip] = static_cast<int>(user_int_parserexec_data[ia](pos.x, pos.y, pos.z, u.x, u.y, u.z, t));
                }
                // Initialize user-defined real attributes with user-defined parser
                for (int ia = 0; ia < n_user_real_attribs; ++ia) {
                    pa_user_real_data[ia][ip] = user_real_parserexec_data[ia](pos.x, pos.y, pos.z, u.x, u.y, u.z, t);
                }

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                // The particle weight is proportional to the user-specified
                // flux and the emission surface within
                // one cell (captured partially by `scale_fac`).
                // For cylindrical emission (flux_normal_axis==0
                // or flux_normal_axis==2), the emission surface depends on
                // the radius ; thus, the calculation is finalized here
                amrex::Real t_weight = flux * scale_fac * dt;
                if (loc_flux_normal_axis != 1) {
                    // Update the weight based on the specified power.
                    // The coefficient ensures that the correct density distribution is obtained.
                    const amrex::Real coeff = 2._rt*MathConst::pi/(1._rt + radial_numpercell_power)
                        *(rmax - std::pow(rmax, -radial_numpercell_power)*std::pow(rmin, 1._rt + radial_numpercell_power));
                    t_weight *= coeff*std::pow(radial_position/rmax, 1._rt - radial_numpercell_power);
                }

                const amrex::Real weight = t_weight;
#elif defined(WARPX_DIM_RSPHERE)
                // The particle weight is proportional to the user-specified
                // flux and the emission surface within
                // one cell (captured partially by `scale_fac`).
                // For spherical emission (flux_normal_axis==0),
                // the emission surface depends on
                // the radius ; thus, the calculation is finalized here
                amrex::Real t_weight = flux * scale_fac * dt;
                if (loc_flux_normal_axis == 0) {
                    // Update the weight based on the specified power.
                    // The coefficient ensures that the correct density distribution is obtained.
                    const amrex::Real coeff = 4._rt*MathConst::pi/(1._rt + radial_numpercell_power)
                        *(rmax*rmax - std::pow(rmax, 1._rt - radial_numpercell_power)*std::pow(rmin, 1._rt + radial_numpercell_power));
                    t_weight *= coeff*std::pow(radial_position/rmax, 2._rt - radial_numpercell_power);
                }
                const amrex::Real weight = t_weight;
#else
                const amrex::Real weight = flux * scale_fac * dt;
#endif
                pa[PIdx::w ][ip] = weight;
                pa[PIdx::ux][ip] = pu.x;
                pa[PIdx::uy][ip] = pu.y;
                pa[PIdx::uz][ip] = pu.z;

                // Update particle position by a random `t_fract`
                // so as to produce a continuous-looking flow of particles
                const amrex::Real t_fract = amrex::Random(engine)*dt;
                UpdatePosition(ppos.x, ppos.y, ppos.z, pu.x, pu.y, pu.z, t_fract);

#if defined(WARPX_DIM_3D)
                pa[PIdx::x][ip] = ppos.x;
                pa[PIdx::y][ip] = ppos.y;
                pa[PIdx::z][ip] = ppos.z;
#elif defined(WARPX_DIM_RZ)
                pa[PIdx::theta][ip] = std::atan2(ppos.y, ppos.x);
                pa[PIdx::x][ip] = std::sqrt(ppos.x*ppos.x + ppos.y*ppos.y);
                pa[PIdx::z][ip] = ppos.z;
#elif defined(WARPX_DIM_XZ)
                pa[PIdx::x][ip] = ppos.x;
                pa[PIdx::z][ip] = ppos.z;
#elif defined(WARPX_DIM_RCYLINDER)
                pa[PIdx::theta][ip] = theta;
                pa[PIdx::x][ip] = radial_position;
#elif defined(WARPX_DIM_RSPHERE)
                pa[PIdx::theta][ip] = theta;
                pa[PIdx::phi][ip] = phi;
                pa[PIdx::x][ip] = radial_position;
#elif defined(WARPX_DIM_1D_Z)
                pa[PIdx::z][ip] = ppos.z;
#endif
            }
        });

        amrex::Gpu::synchronize();

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

    // Remove particles that are inside the embedded boundaries
#ifdef AMREX_USE_EB
    if (EB::enabled())
    {
        using warpx::fields::FieldType;
        auto & warpx = WarpX::GetInstance();
        scrapeParticlesAtEB(
            tmp_pc,
            warpx.m_fields.get_mr_levels(FieldType::distance_to_eb, warpx.finestLevel()),
            ParticleBoundaryProcess::Absorb());
    }
#endif

    // Redistribute the new particles that were added to the temporary container.
    // (This eliminates invalid particles, and makes sure that particles
    // are in the right tile.)
    tmp_pc.Redistribute();

    // Add the particles to the current container
    this->addParticles(tmp_pc, true);
}
