/* Copyright 2019-2020 Luca Fedeli, Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_SMART_CREATE_H_
#define WARPX_SMART_CREATE_H_

#include "DefaultInitialization.H"

#include <AMReX.H>
#include <AMReX_AmrCore.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Particle.H>
#include <AMReX_ParticleTile.H>

/**
 * \brief This is a functor for performing a "smart create" that works
 * in both host and device code.
 *
 * A "smart" create does the following:
 * First, it initializes the position, cpuid and id of
 * the particle (in most cases IDs should be initialized using setNewParticleIDs
 * after their creation).
 * Then, it initializes all the other components according to initialization policies.
 *
 * You don't create this directly - use the SmartCreateFactory object below.
 *
 */

struct SmartCreate
{
    const InitializationPolicy* m_policy_real;
    const InitializationPolicy* m_policy_int;
    int m_weight_index = 0;

    template <typename PartData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (
        PartData& prt, const int i_prt,
        amrex::RandomEngine const& engine,
        const amrex::Real x = 0.0,
        const amrex::Real y = 0.0,
        const amrex::Real z = 0.0,
        const int cpu = 0,
        const int id = 0) const noexcept
    {
#if defined(WARPX_DIM_3D)
        prt.rdata(PIdx::x)[i_prt] = x;
        prt.rdata(PIdx::y)[i_prt] = y;
        prt.rdata(PIdx::z)[i_prt] = z;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        prt.rdata(PIdx::x)[i_prt] = x;
        prt.rdata(PIdx::z)[i_prt] = z;
        amrex::ignore_unused(y);
#elif defined(WARPX_DIM_1D_Z)
        prt.rdata(PIdx::z)[i_prt] = z;
        amrex::ignore_unused(x,y);
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        prt.rdata(PIdx::x)[i_prt] = x;
        amrex::ignore_unused(y,z);
#endif

        prt.m_idcpu[i_prt] = amrex::SetParticleIDandCPU(id, cpu);

        // initialize the real components after position
        for (int j = AMREX_SPACEDIM; j < prt.m_n_real; ++j) {
            prt.rdata(j)[i_prt] = initializeRealValue(m_policy_real[j], engine);
        }

        // initialize the int components
        for (int j = 0; j < prt.m_n_int; ++j) {
            prt.idata(j)[i_prt] = initializeIntValue(m_policy_int[j]);
        }
    }
};

/**
 * \brief A factory for creating SmartCreate functors.
 *
 * Given a particle container, this can create a functor
 * that will perform the smart create operation on a tile
 * of that particle container
 */
class SmartCreateFactory
{
    PolicyVec m_policy_real;
    PolicyVec m_policy_int;
    bool m_defined{false};

public:
    template <class PartTileData>
    SmartCreateFactory (const PartTileData& part) noexcept:
        m_policy_real{getPolicies(part.GetRealSoANames())},
        m_policy_int{getPolicies(part.GetIntSoANames())},
        m_defined{true}
    {}

    [[nodiscard]] SmartCreate getSmartCreate () const noexcept
    {
        AMREX_ASSERT(m_defined);
        return SmartCreate{m_policy_real.dataPtr(),
                         m_policy_int.dataPtr()};
    }

    [[nodiscard]] bool isDefined () const noexcept { return m_defined; }
};

#endif //WARPX_SMART_CREATE_H_
