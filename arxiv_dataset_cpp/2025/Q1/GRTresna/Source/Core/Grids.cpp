/* GRTresna
 * Copyright 2024 The GRTL Collaboration.
 * Please refer to LICENSE in GRTresna's root directory.
 */

#include "Grids.hpp"

#include "AMRIO.H"
#include "BRMeshRefine.cpp"
#include "CoarseAverage.H"
#include "FilesystemTools.hpp"
#include "FourthOrderCFInterp.H"
#include "GRParmParse.hpp"
#include "IntVectSet.H"
#include "LoadBalance.H"
#include "ProblemDomain.H"
#include "REAL.H"
#include "RealVect.H"
#include "VariableCoeffPoissonOperatorFactory.H"

void Grids::read_params(GRParmParse &pp, params_t &m_grid_params)
{
    // Chombo grid params
    pp.get("max_level", m_grid_params.maxLevel);
    m_grid_params.numLevels = m_grid_params.maxLevel + 1;
    pout() << "Num levels read out is " << m_grid_params.numLevels << endl;
    std::vector<int> nCellsArray(SpaceDim);
    pp.getarr("N", nCellsArray, 0, SpaceDim);
    for (int idir = 0; idir < SpaceDim; idir++)
    {
        m_grid_params.nCells[idir] = nCellsArray[idir];
    }
    // Enforce that dx is same in every directions
    // and that ref_ratio = 2 always as these conditions
    // are required in several places in our code
    m_grid_params.refRatio.resize(m_grid_params.numLevels);
    m_grid_params.refRatio.assign(2);
    Real domain_length;
    pp.get("L", domain_length);
    int max_cells = max(m_grid_params.nCells[0], m_grid_params.nCells[1]);
    max_cells = max(m_grid_params.nCells[2], max_cells);
    m_grid_params.coarsestDx = domain_length / max_cells;
    for (int idir = 0; idir < SpaceDim; idir++)
    {
        m_grid_params.domainLength[idir] =
            m_grid_params.coarsestDx * m_grid_params.nCells[idir];
    }

    // Chombo refinement and load balancing criteria
    pp.load("refine_threshold", m_grid_params.refineThresh, 0.5);
    pp.load("block_factor", m_grid_params.blockFactor, 16);
    pp.load("max_grid_size", m_grid_params.maxGridSize, 16);
    pp.load("fill_ratio", m_grid_params.fillRatio, 0.75);
    pp.load("buffer_size", m_grid_params.bufferSize, 0);
    pp.load("regrid_radius", m_grid_params.regrid_radius, 0.0);

    // Default number of ghosts
    m_grid_params.num_ghosts = 3;

    // how to average face-centered coefficients to coarser multigrid levels
    // default is harmonic
    m_grid_params.coefficient_average_type = CoarseAverage::harmonic;
    ;
    if (pp.contains("coefficient_average_type"))
    {
        std::string tempString;
        pp.get("coefficient_average_type", tempString);
        if (tempString == "arithmetic")
        {
            m_grid_params.coefficient_average_type = CoarseAverage::arithmetic;
        }
        else if (tempString == "harmonic")
        {
            m_grid_params.coefficient_average_type = CoarseAverage::harmonic;
        }
        else
        {
            MayDay::Error("bad coefficient_average_type in input");
        }
    } // end if an average_type is present in inputs

    // set up coarse domain box
    IntVect lo = IntVect::Zero;
    IntVect hi = m_grid_params.nCells;
    hi -= IntVect::Unit;
    Box crseDomBox(lo, hi);
    m_grid_params.probLo = RealVect::Zero;
    m_grid_params.probHi = RealVect::Zero;
    m_grid_params.probHi += m_grid_params.domainLength;

    // Periodicity
    ProblemDomain crseDom(crseDomBox);
    m_grid_params.periodic.resize(SpaceDim);
    pp.getarr("is_periodic", m_grid_params.periodic, 0, SpaceDim);
    m_grid_params.periodic_directions_exist = false;
    for (int dir = 0; dir < SpaceDim; dir++)
    {
        crseDom.setPeriodic(dir, m_grid_params.periodic[dir]);
        if (m_grid_params.periodic[dir])
        {
            m_grid_params.periodic_directions_exist = true;
        }
    }
    m_grid_params.coarsestDomain = crseDom;

    // now the boundary read in
    m_grid_params.boundary_params.read_params(pp);

    FOR1(idir)
    {
        // default center to center of grid, may reset below
        // depending on boundaries
        m_grid_params.center[idir] = domain_length / 2.0;

        if ((m_grid_params.boundary_params.lo_boundary[idir] ==
             BoundaryConditions::REFLECTIVE_BC) &&
            (m_grid_params.boundary_params.hi_boundary[idir] !=
             BoundaryConditions::REFLECTIVE_BC))
            m_grid_params.center[idir] = 0.;
        else if ((m_grid_params.boundary_params.hi_boundary[idir] ==
                  BoundaryConditions::REFLECTIVE_BC) &&
                 (m_grid_params.boundary_params.lo_boundary[idir] !=
                  BoundaryConditions::REFLECTIVE_BC))
            m_grid_params.center[idir] = domain_length;
        else if ((m_grid_params.boundary_params.hi_boundary[idir] ==
                  BoundaryConditions::REFLECTIVE_BC) &&
                 (m_grid_params.boundary_params.lo_boundary[idir] ==
                  BoundaryConditions::REFLECTIVE_BC))
            m_grid_params.center[idir] = 0.;
    }

    pout() << "grid center set to " << m_grid_params.center[0] << " "
           << m_grid_params.center[1] << " " << m_grid_params.center[2] << endl;
}

void Grids::read_grids(std::string input_filename)
{
#ifdef CH_USE_HDF5

    // set up a temp data structure for the source data
    Vector<LevelData<FArrayBox> *> temp_data;
    Vector<string> variable_names;
    int temp_num_levels;
    Real dx, dt, time;
    Box temp_domain_box;
    Vector<int> temp_ref_ratio;

    // update directly the input grids
    ReadAMRHierarchyHDF5(input_filename, grids_data, temp_data, variable_names,
                         temp_domain_box, dx, dt, time, temp_ref_ratio,
                         temp_num_levels);

    int max_temp_level = temp_data.size();
    // clean up temporary storage
    for (int level = 0; level < max_temp_level; level++)
    {
        delete temp_data[level];
        temp_data[level] = NULL;
    }
#endif
}

void Grids::define_operator(MultilevelLinearOp<FArrayBox> &mlOp,
                            Vector<RefCountedPtr<LevelData<FArrayBox>>> &aCoef,
                            Vector<RefCountedPtr<LevelData<FArrayBox>>> &bCoef,
                            const Real &a_alpha, const Real &a_beta)
{
    RefCountedPtr<AMRLevelOpFactory<LevelData<FArrayBox>>> opFactory =
        RefCountedPtr<AMRLevelOpFactory<LevelData<FArrayBox>>>(
            defineOperatorFactory(grids_data, vectDomain, aCoef, bCoef,
                                  m_grid_params, a_alpha, a_beta));

    int lBase = 0;
    mlOp.define(grids_data, m_grid_params.refRatio, vectDomain, vectDx,
                opFactory, lBase);
}

void Grids::update_psi0(Vector<LevelData<FArrayBox> *> multigrid_vars,
                        Vector<LevelData<FArrayBox> *> constraint_vars,
                        bool deactivate_zero_mode)
{
    for (int ilev = 0; ilev < m_grid_params.numLevels; ilev++)
    {
        IntVect ghosts = m_grid_params.num_ghosts * IntVect::Unit;
        // For interlevel ghosts in constraint_vars
        if (ilev > 0)
        {
            // Fill using 4th order method which fills corner ghosts
            int num_ghosts = 1;
            FourthOrderCFInterp m_patcher;
            m_patcher.define(grids_data[ilev], grids_data[ilev - 1],
                             NUM_CONSTRAINT_VARS, vectDomain[ilev - 1],
                             m_grid_params.refRatio[ilev], num_ghosts);
            m_patcher.coarseFineInterp(*constraint_vars[ilev],
                                       *constraint_vars[ilev - 1], 0, 0,
                                       NUM_CONSTRAINT_VARS);
        }

        // For intralevel ghosts - this is done below
        // but need the exchange copier object to do this
        BoundaryConditions solver_boundaries;
        solver_boundaries.define(vectDx[ilev][0], m_grid_params.boundary_params,
                                 vectDomain[ilev], m_grid_params.num_ghosts);
        // For intralevel ghosts
        DisjointBoxLayout grown_grids;
        solver_boundaries.expand_grids_to_boundaries(grown_grids,
                                                     grids_data[ilev]);
        Copier exchange_copier;
        exchange_copier.exchangeDefine(grown_grids, ghosts);

        // now the update

        // first exchange ghost cells for constraint_vars so they are filled
        // with the correct values
        constraint_vars[ilev]->exchange(constraint_vars[ilev]->interval(),
                                        exchange_copier);

        DataIterator dit = multigrid_vars[ilev]->dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        {
            FArrayBox &multigrid_vars_box = (*multigrid_vars[ilev])[dit()];
            FArrayBox &constraint_vars_box = (*constraint_vars[ilev])[dit()];

            Box ghosted_box = multigrid_vars_box.box();
            BoxIterator bit(ghosted_box);
            for (bit.begin(); bit.ok(); ++bit)
            {
                IntVect iv = bit();

                // Update constraint variables for the linear step
                multigrid_vars_box(iv, c_psi_reg) +=
                    constraint_vars_box(iv, c_psi);
                if (deactivate_zero_mode)
                {
                    multigrid_vars_box(iv, c_V1_0) +=
                        constraint_vars_box(iv, c_V1);
                    multigrid_vars_box(iv, c_V2_0) +=
                        constraint_vars_box(iv, c_V2);
                    multigrid_vars_box(iv, c_V3_0) +=
                        constraint_vars_box(iv, c_V3);
                    multigrid_vars_box(iv, c_U_0) +=
                        constraint_vars_box(iv, c_U);
                }
                else
                {
                    multigrid_vars_box(iv, c_V1_0) =
                        constraint_vars_box(iv, c_V1);
                    multigrid_vars_box(iv, c_V2_0) =
                        constraint_vars_box(iv, c_V2);
                    multigrid_vars_box(iv, c_V3_0) =
                        constraint_vars_box(iv, c_V3);
                    multigrid_vars_box(iv, c_U_0) =
                        constraint_vars_box(iv, c_U);
                }
            }
        }
    }
}

void Grids::fill_ghosts_correct_coarse(
    Vector<LevelData<FArrayBox> *> multigrid_vars, bool filling_solver_vars)
{
    for (int ilev = 0; ilev < m_grid_params.numLevels; ilev++)
    {
        // fill the boundary cells and ghosts
        BoundaryConditions solver_boundaries;
        solver_boundaries.define(vectDx[ilev][0], m_grid_params.boundary_params,
                                 vectDomain[ilev], m_grid_params.num_ghosts);

        const Interval &a_comps = filling_solver_vars
                                      ? Interval(c_psi_reg, c_U_0)
                                      : Interval(0, NUM_MULTIGRID_VARS - 1);

        // this will populate the multigrid boundaries according to the BCs
        // in particular it will fill cells for Aij, and updated K
        solver_boundaries.fill_multigrid_boundaries(
            Side::Lo, *multigrid_vars[ilev], a_comps,
            filling_solver_vars); //, Interval(c_K_0, c_A33_0));
        solver_boundaries.fill_multigrid_boundaries(
            Side::Hi, *multigrid_vars[ilev], a_comps,
            filling_solver_vars); //, Interval(c_K_0, c_A33_0));

        // To define an exchange copier to cover the outer ghosts
        DisjointBoxLayout grown_grids;
        solver_boundaries.expand_grids_to_boundaries(grown_grids,
                                                     grids_data[ilev]);

        // Fill the interlevel ghosts from a coarser level
        if (ilev > 0)
        {
            QuadCFInterp quadCFI(grids_data[ilev], &grids_data[ilev - 1],
                                 vectDx[ilev][0], m_grid_params.refRatio[ilev],
                                 NUM_MULTIGRID_VARS, vectDomain[ilev]);
            quadCFI.coarseFineInterp(*multigrid_vars[ilev],
                                     *multigrid_vars[ilev - 1]);
        }

        // exchange the interior ghosts
        IntVect ghosts = m_grid_params.num_ghosts * IntVect::Unit;
        Copier exchange_copier;
        exchange_copier.exchangeDefine(grown_grids, ghosts);
        multigrid_vars[ilev]->exchange(multigrid_vars[ilev]->interval(),
                                       exchange_copier);
    }
}

void Grids::set_grids()
{
    set_domains_and_dx(vectDomain, vectDx);

    int numlevels = m_grid_params.numLevels;

    ParmParse pp;

    // grid generation parameters
    grids_data.resize(numlevels);

    int maxLevel = numlevels - 1;
    Vector<Vector<Box>> newBoxes(numlevels);
    Vector<Vector<Box>> oldBoxes(numlevels);

    // determine grids dynamically, based on the tagging criteria
    Vector<LevelData<FArrayBox> *> vect_tagging_criterion(maxLevel + 1, NULL);

    // define base level first
    Vector<Vector<int>> procAssign(maxLevel + 1);
    domainSplit(vectDomain[0], oldBoxes[0], m_grid_params.maxGridSize,
                m_grid_params.blockFactor);
    procAssign[0].resize(oldBoxes[0].size());
    LoadBalance(procAssign[0], oldBoxes[0]);
    grids_data[0].define(oldBoxes[0], procAssign[0], vectDomain[0]);
    vect_tagging_criterion[0] = new LevelData<FArrayBox>(
        grids_data[0], 1, // only one value in this array
        IntVect::Zero);

    int topLevel = 0;
    bool moreLevels = (maxLevel > 0);

    int nesting_radius = 2;
    // create grid generation object
    BRMeshRefine meshrefine(vectDomain[0], m_grid_params.refRatio,
                            m_grid_params.fillRatio, m_grid_params.blockFactor,
                            nesting_radius, m_grid_params.maxGridSize);

    while (moreLevels)
    {
        // default is moreLevels = false
        // (only repeat loop in the case where a new level
        // is generated which is still less than maxLevel)
        moreLevels = false;

        int baseLevel = 0;
        int oldTopLevel = topLevel;
        // now initialize tagging criterion for this existing hierarchy
        for (int level = 0; level <= topLevel; level++)
        {
            RealVect dxLevel = vectDx[level];

            LevelData<FArrayBox> *temp_multigrid_vars;

            IntVect ghosts = 1 * IntVect::Unit;
            temp_multigrid_vars = new LevelData<FArrayBox>(
                grids_data[level], NUM_MULTIGRID_VARS, ghosts);

            tagging_criterion->set_regrid_condition(
                *vect_tagging_criterion[level], *temp_multigrid_vars, dxLevel,
                m_grid_params.center, m_grid_params.regrid_radius);

            if (temp_multigrid_vars != NULL)
            {
                delete temp_multigrid_vars;
                temp_multigrid_vars = NULL;
            }
        }
        Vector<IntVectSet> tagVect(topLevel + 1);
        int tags_grow = 2;
        set_tag_cells(vect_tagging_criterion, tagVect, vectDx, vectDomain,
                      tags_grow, baseLevel, topLevel + 1);

        int new_finest =
            meshrefine.regrid(newBoxes, tagVect, baseLevel, topLevel, oldBoxes);

        if (new_finest > topLevel)
        {
            topLevel++;
        }
        oldBoxes = newBoxes;

        //  no need to do this for the base level (already done)
        for (int lev = 1; lev <= topLevel; lev++)
        {
            // do load balancing
            procAssign[lev].resize(newBoxes[lev].size());
            LoadBalance(procAssign[lev], newBoxes[lev]);
            const DisjointBoxLayout newDBL(newBoxes[lev], procAssign[lev],
                                           vectDomain[lev]);
            grids_data[lev] = newDBL;
            delete vect_tagging_criterion[lev];
            vect_tagging_criterion[lev] = new LevelData<FArrayBox>(
                grids_data[lev], 1, IntVect::Zero); // again only one entry
        } // end loop over levels for initialization
        // figure out whether we need another pass through grid generation
        if ((topLevel < maxLevel) && (topLevel > oldTopLevel))
        {
            moreLevels = true;
        }
        else
        {
            break;
        }
    } // end while moreLevels loop

    // clean up temp storage
    for (int ilev = 0; ilev < vect_tagging_criterion.size(); ilev++)
    {
        if (vect_tagging_criterion[ilev] != NULL)
        {
            delete vect_tagging_criterion[ilev];
            vect_tagging_criterion[ilev] = NULL;
        }
    }
}

void Grids::set_domains_and_dx(Vector<ProblemDomain> &vectDomain,
                               Vector<RealVect> &vectDx)
{

    vectDomain.resize(m_grid_params.numLevels);
    vectDx.resize(m_grid_params.numLevels);
    vectDx[0] = m_grid_params.coarsestDx * RealVect::Unit;
    for (int ilev = 1; ilev < m_grid_params.numLevels; ilev++)
    {
        vectDx[ilev] = vectDx[ilev - 1] / m_grid_params.refRatio[ilev - 1];
    }

    vectDomain[0] = m_grid_params.coarsestDomain;
    for (int ilev = 1; ilev < m_grid_params.numLevels; ilev++)
    {
        vectDomain[ilev] =
            refine(vectDomain[ilev - 1], m_grid_params.refRatio[ilev - 1]);
    }
}

void Grids::set_tag_cells(
    Vector<LevelData<FArrayBox> *> &vect_tagging_criterion,
    Vector<IntVectSet> &tagVect, Vector<RealVect> &vectDx,
    Vector<ProblemDomain> &vectDomain, const int tags_grow, const int baseLevel,
    int numLevels_tag)
{
    for (int lev = baseLevel; lev != numLevels_tag; lev++)
    {
        IntVectSet local_tags;
        LevelData<FArrayBox> &level_tagging_criterion =
            *vect_tagging_criterion[lev];
        DisjointBoxLayout level_domain = level_tagging_criterion.getBoxes();
        DataIterator dit = level_tagging_criterion.dataIterator();

        Real max_tagging_criterion = 0;
        max_tagging_criterion = norm(level_tagging_criterion,
                                     level_tagging_criterion.interval(), 0);
        Real tagVal = max_tagging_criterion * m_grid_params.refineThresh;
        // now loop through grids and tag cells where tagging crierion > tagVal
        for (dit.reset(); dit.ok(); ++dit)
        {
            const Box thisBox = level_domain.get(dit());
            const FArrayBox &this_tagging_criterion =
                level_tagging_criterion[dit()];
            BoxIterator bit(thisBox);
            for (bit.begin(); bit.ok(); ++bit)
            {
                const IntVect &iv = bit();
                if (abs(this_tagging_criterion(iv)) >= tagVal)
                    local_tags |= iv;
            }
        } // end loop over grids on this level
        local_tags.grow(tags_grow);
        const Box &domainBox = vectDomain[lev].domainBox();
        local_tags &= domainBox;

        tagVect[lev] = local_tags;

    } // end loop over levels
}

void Grids::get_loc(RealVect &a_out_loc, const IntVect &a_iv,
                    const RealVect &a_dx,
                    const std::array<double, SpaceDim> center)
{
    a_out_loc = a_iv + 0.5 * RealVect::Unit;
    a_out_loc *= a_dx;
    FOR1(i) { a_out_loc[i] -= center[i]; }
}
