/* Copyright 2019-2020 Andrew Myers, Axel Huebl,
 *                     Maxence Thevenet
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SMART_COPY_H_
#define WARPX_SMART_COPY_H_

#include "DefaultInitialization.H"
#include "SmartUtils.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>

#include <map>
#include <string>

/**
 * \brief This is a functor for performing a "smart copy" that works
 * in both host and device code.
 *
 * A "smart" copy does the following. First, the destination particle
 * components are initialized to the default values for that component
 * type. Second, if a given component name is found in both the src
 * and the dst, then the src value is copied.
 *
 * Particle positions and id numbers are always copied.
 *
 * You don't create this directly - use the SmartCopyFactory object below.
 */
struct SmartCopy
{
    int m_num_copy_real;
    const int* m_src_comps_r;
    const int* m_dst_comps_r;

    int m_num_copy_int;
    const int* m_src_comps_i;
    const int* m_dst_comps_i;

    const InitializationPolicy* m_policy_real;
    const InitializationPolicy* m_policy_int;

    template <typename DstData, typename SrcData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (DstData& dst, const SrcData& src, int i_src, int i_dst,
                     amrex::RandomEngine const& engine) const noexcept
    {
        // initialize the real components
        for (int j = 0; j < dst.m_n_real; ++j) {
            dst.rdata(j)[i_dst] = initializeRealValue(m_policy_real[j], engine);
        }

        // initialize the int components
        for (int j = 0; j < dst.m_n_int; ++j) {
            dst.idata(j)[i_dst] = initializeIntValue(m_policy_int[j]);
        }

        // copy the shared real components
        for (int j = 0; j < m_num_copy_real; ++j)
        {
            int src_comp, dst_comp;
            amrex::ParticleReal* AMREX_RESTRICT dst_data;
            const amrex::ParticleReal* AMREX_RESTRICT src_data;

            src_comp = m_src_comps_r[j];
            src_data = src.rdata(src_comp);

            dst_comp = m_dst_comps_r[j];
            dst_data = dst.rdata(dst_comp);

            dst_data[i_dst] = src_data[i_src];
        }

        // copy the shared int components
        for (int j = 0; j < m_num_copy_int; ++j)
        {
            int src_comp, dst_comp;
            int* AMREX_RESTRICT dst_data;
            const int* AMREX_RESTRICT src_data;

            src_comp = m_src_comps_i[j];
            src_data = src.idata(src_comp);

            dst_comp = m_dst_comps_i[j];
            dst_data = dst.idata(dst_comp);

            dst_data[i_dst] = src_data[i_src];
        }
    }
};

/**
 * \brief A factory for creating SmartCopy functors.
 *
 * Given two particle containers, this can create a functor
 * that will perform the smart copy operation between those
 * particle container's tiles.
 */
class SmartCopyFactory
{
    SmartCopyTag m_tag_real;
    SmartCopyTag m_tag_int;
    PolicyVec m_policy_real;
    PolicyVec m_policy_int;
    bool m_defined = false;

public:
    template <class SrcPC, class DstPC>
    SmartCopyFactory (const SrcPC& src, const DstPC& dst) noexcept :
        m_tag_real{getSmartCopyTag(src.GetRealSoANames(), dst.GetRealSoANames())},
        m_tag_int{getSmartCopyTag(src.GetIntSoANames(), dst.GetIntSoANames())},
        m_policy_real{getPolicies(dst.GetRealSoANames())},
        m_policy_int{getPolicies(dst.GetIntSoANames())},
        m_defined{true}
    {}

    [[nodiscard]] SmartCopy getSmartCopy () const noexcept
    {
        AMREX_ASSERT(m_defined);
        return SmartCopy{m_tag_real.size(),
                         m_tag_real.src_comps.dataPtr(),
                         m_tag_real.dst_comps.dataPtr(),
                         m_tag_int.size(),
                         m_tag_int. src_comps.dataPtr(),
                         m_tag_int. dst_comps.dataPtr(),
                         m_policy_real.dataPtr(),
                         m_policy_int.dataPtr()};
    }

    [[nodiscard]] bool isDefined () const noexcept { return m_defined; }
};

#endif //WARPX_SMART_COPY_H_
