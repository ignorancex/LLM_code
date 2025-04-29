#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "core_allvars.h"
#include "core_proto.h"


void reincorporate_gas(int p, int centralgal, double dt)
{
    double move_factor;
    double sat_rad = get_satellite_radius(p, centralgal);
    const double t_dyn = 0.1 / sqrt(Hubble_sqr_z(Halo[Gal[centralgal].HaloNr].SnapNum));
    assert(Gal[p].EjectedMass >= Gal[p].MetalsEjectedMass);
    assert(Gal[p].OutflowGas >= 0.0);

    // reincorporate the fountaining gas
    if(Gal[p].FountainGas > 0.0)
    {
        if(dt >= Gal[p].FountainTime)
        {
            Gal[p].HotGas += Gal[p].FountainGas;
            Gal[p].MetalsHotGas += Gal[p].MetalsFountainGas;
            assert(Gal[p].MetalsHotGas<=Gal[p].HotGas);
            Gal[p].FountainGas = 0.0;
            Gal[p].MetalsFountainGas = 0.0;
            Gal[p].FountainTime = 0.0;
        }
        else
        {
            if(!(Gal[p].FountainTime > 0.0))
            {
                printf("dt, Gal[p].FountainTime, Gal[p].FountainGas = %e, %e, %e\n", dt, Gal[p].FountainTime, Gal[p].FountainGas);
            }
            assert(Gal[p].FountainTime > 0.0);
            
            move_factor = dt / Gal[p].FountainTime;
            Gal[p].HotGas += (Gal[p].FountainGas * move_factor);
            Gal[p].MetalsHotGas += (Gal[p].MetalsFountainGas * move_factor);
            assert(Gal[p].MetalsHotGas<=Gal[p].HotGas);
            Gal[p].FountainGas -= (Gal[p].FountainGas * move_factor);
            Gal[p].MetalsFountainGas -= (Gal[p].MetalsFountainGas * move_factor);
            Gal[p].FountainTime -= dt;
        }
    }
    
    
    // recincorporate inflowing gas from the ejected reservoir
    if(Gal[p].EjectedMass > 0.0)
    {
        // use time-scale calculated in the code (calculated/modified below and when the halo grows)
        if(dt >= Gal[p].ReincTime)
        {
            if(Gal[p].FountainGas + Gal[p].EjectedMass > 0.0)
                Gal[p].FountainTime = (Gal[p].FountainGas * Gal[p].FountainTime + Gal[p].EjectedMass * t_dyn) / (Gal[p].FountainGas + Gal[p].EjectedMass);
            Gal[p].FountainGas += Gal[p].EjectedMass;
            Gal[p].MetalsFountainGas += Gal[p].MetalsEjectedMass;
            Gal[p].EjectedMass = 0.0;
            Gal[p].MetalsEjectedMass = 0.0;
            Gal[p].ReincTime = 0.0; // perhaps should be set to the dynamical time as a default instead, but should make no difference with zero mass
        }
        else
        {
            move_factor = dt / Gal[p].ReincTime;
            if(Gal[p].FountainGas + move_factor*Gal[p].EjectedMass > 0.0)
                Gal[p].FountainTime = (Gal[p].FountainGas * Gal[p].FountainTime + move_factor*Gal[p].EjectedMass * t_dyn) / (Gal[p].FountainGas + move_factor*Gal[p].EjectedMass);
            Gal[p].FountainGas += (Gal[p].EjectedMass * move_factor);
            Gal[p].MetalsFountainGas += (Gal[p].MetalsEjectedMass * move_factor);
            Gal[p].EjectedMass -= (Gal[p].EjectedMass * move_factor);
            Gal[p].MetalsEjectedMass -= (Gal[p].MetalsEjectedMass * move_factor);
            
            if(!(Gal[p].EjectedMass >= Gal[p].MetalsEjectedMass))
            {
                printf("Gal[p].EjectedMass, Gal[p].MetalsEjectedMass = %e, %e\n", Gal[p].EjectedMass, Gal[p].MetalsEjectedMass);
                printf("move_factor, Gal[p].ReincTime = %e, %e\n", move_factor, Gal[p].ReincTime);
            }
            
            assert(Gal[p].EjectedMass >= Gal[p].MetalsEjectedMass);
            Gal[p].ReincTime -= dt;
            assert(Gal[p].ReincTime >= 0.0 && Gal[p].ReincTime==Gal[p].ReincTime);
        }
    }
    
    
    // put outflowing gas into the ejected reservoir (or the central's fountain reservoir if appropriate)
    if(Gal[p].OutflowGas > 0.0)
    {
        if(Gal[p].OutflowTime > 0.0)
            move_factor = dt / Gal[p].OutflowTime;
        else
            move_factor = 1.0;
        const double eject_gas = Gal[p].OutflowGas * dmin(1.0, move_factor);
        assert(eject_gas >= 0.0);
        assert(eject_gas <= Gal[p].OutflowGas);
        
        if(p==centralgal || sat_rad > Gal[centralgal].Rvir)
        {
            // want to energy to calculate where the gas that is ejected gets to before it turns around
            // first need to work out its initial speed
            // not considering any tangential motion anymore
            double outflow_kinetic, outflow_radial_speed; //j_hot, discriminant, ;
            const double thermal = 0.5 * sqr(Gal[p].Vvir);
            outflow_kinetic = Gal[p].OutflowSpecificEnergy - Gal[p].EjectedPotential - thermal;           
            
            if(outflow_kinetic > 0.0)
                outflow_radial_speed = sqrt(2.0 * outflow_kinetic);
            else
                outflow_radial_speed = 0.0;
            
            // iterate until the new turnaround radius is found based on the ejected gas's potential
            double R_min, R_max, R_guess, pot_guess, specific_energy;
            int iter;
            R_min = Gal[p].Rvir;
            R_max = 10 * Gal[p].Rvir;
            for(iter=0; iter<200; iter++)
            {
                R_guess = 0.5 * (R_min + R_max);
                pot_guess = NFW_potential(p, R_guess); // approximating an NFW potential here for efficiency
                specific_energy = pot_guess + thermal; // add thermal and tangential kinetic energy assuming angular momentum is conserved
                
                if(fabs((specific_energy-Gal[p].OutflowSpecificEnergy)/Gal[p].OutflowSpecificEnergy) < 1e-3)
                    break;
                
                if(specific_energy > Gal[p].OutflowSpecificEnergy)
                    R_max = R_guess;
                else
                    R_min = R_guess;
            }

            // assume time to reincorporate the freshly outflowing gas is that to get to the turnaround point and back, assuming its average speed is half its outflow speed
            double reinc_time_new;
            if(outflow_radial_speed > 0.0)
                reinc_time_new = 4.0 * (R_guess - Gal[p].Rvir) / outflow_radial_speed;
            else
                reinc_time_new = 0.0;

            assert(Gal[p].ReincTime >= 0 && Gal[p].ReincTime == Gal[p].ReincTime);

            if(eject_gas > 0.0 && reinc_time_new > 0.0)
            {
                Gal[p].ReincTime = (Gal[p].EjectedMass * Gal[p].ReincTime + eject_gas * reinc_time_new) / (Gal[p].EjectedMass + eject_gas);
                Gal[p].EjectedSpecificEnergy = (Gal[p].EjectedMass * Gal[p].EjectedSpecificEnergy + eject_gas * specific_energy) / (Gal[p].EjectedMass + eject_gas);
            }
            
            assert(Gal[p].ReincTime >= 0 && Gal[p].ReincTime == Gal[p].ReincTime);
            
            if(reinc_time_new <= 0.0)
            { // if this isn't positive, then the outflowing material immediately turns around in effect
                if(Gal[p].FountainGas + Gal[p].OutflowGas > 0)
                    Gal[p].FountainTime = (Gal[p].FountainGas * Gal[p].FountainTime + Gal[p].OutflowGas * t_dyn) / (Gal[p].FountainGas + Gal[p].OutflowGas);
                Gal[p].FountainGas += Gal[p].OutflowGas;
                Gal[p].MetalsFountainGas += Gal[p].MetalsOutflowGas;
                Gal[p].OutflowGas = 0.0;
                Gal[p].MetalsOutflowGas = 0.0;
                Gal[p].OutflowTime = 0.0;
            }
            else if(move_factor >= 1.0)
            {
                Gal[p].EjectedMass += Gal[p].OutflowGas;
                Gal[p].MetalsEjectedMass += Gal[p].MetalsOutflowGas;
                assert(Gal[p].EjectedMass >= Gal[p].MetalsEjectedMass);

                Gal[p].OutflowGas = 0.0;
                Gal[p].MetalsOutflowGas = 0.0;
                Gal[p].OutflowTime = 0.0;
            }
            else
            {
                Gal[p].EjectedMass += eject_gas;
                Gal[p].MetalsEjectedMass += (Gal[p].MetalsOutflowGas * move_factor);
                assert(Gal[p].EjectedMass >= Gal[p].MetalsEjectedMass);

                assert(Gal[p].OutflowGas >= 0.0);
                if(Gal[p].OutflowGas - eject_gas < 0.0) printf("Gal[p].OutflowGas, Gal[p].OutflowTime = %e, %e\n", Gal[p].OutflowGas, Gal[p].OutflowTime);
                Gal[p].OutflowGas -= eject_gas;
                Gal[p].MetalsOutflowGas -= (Gal[p].MetalsOutflowGas * move_factor);
                Gal[p].OutflowTime -= dt;
                assert(eject_gas >= 0.0);
                if(!(Gal[p].OutflowGas >= 0.0))
                {
                    printf("eject_gas, Gal[p].OutflowGas, move_factor, Gal[p].OutflowTime, dt = %e, %e, %e, %e, %e\n", eject_gas, Gal[p].OutflowGas, move_factor, Gal[p].OutflowTime, dt);
                }
                assert(Gal[p].OutflowGas >= 0.0);
            }
        }
        else
        {
            // satellite internal to Rvir, so outflowing gas goes to central's fountain reservoir instead
            assert(Gal[centralgal].FountainGas + eject_gas > 0.0);
            Gal[centralgal].FountainTime = (Gal[centralgal].FountainGas * Gal[centralgal].FountainTime + eject_gas * t_dyn) / (Gal[centralgal].FountainGas + eject_gas);
            
            if(dt >= Gal[p].OutflowTime)
            {
                Gal[centralgal].FountainGas += Gal[p].OutflowGas;
                Gal[centralgal].MetalsFountainGas += Gal[p].MetalsOutflowGas;
                Gal[p].OutflowGas = 0.0;
                Gal[p].MetalsOutflowGas = 0.0;
                Gal[p].OutflowTime = 0.0;
            }
            else
            {
                Gal[centralgal].FountainGas += eject_gas;
                Gal[centralgal].MetalsFountainGas += (Gal[p].MetalsOutflowGas * move_factor);
                assert(Gal[p].OutflowGas >= 0.0);
                Gal[p].OutflowGas -= eject_gas;
                Gal[p].MetalsOutflowGas -= (Gal[p].MetalsOutflowGas * move_factor);
                Gal[p].OutflowTime -= dt;
                assert(Gal[p].OutflowGas >= 0.0);
            }
            
        }
        
    }
    
}


void update_galactic_fountain_from_growth(int p)
{ // p should be a central galaxy
    
    // no point doing this function if the halo has not actually grown!
    if(Gal[p].Rvir <= Gal[p].prevRvir) return;

    
    double reinc_mass, metallicity;
    const double t_dyn = Gal[p].Rvir / Gal[p].Vvir;
    
    // quantity same for hot, fountain, and outflow reservoirs
    const double thermal = 0.5 * sqr(Gal[p].Vvir);
    const double potential_and_thermal = Gal[p].Potential[0] + thermal;
    const double j_hot = 2.0 * Gal[p].Vvir * Gal[p].CoolScaleRadius;
    const double hot_specific_energy = Gal[p].HotGasPotential + thermal + 0.5*sqr(j_hot)/Gal[p].R2_hot_av;
    
    if(Gal[p].OutflowGas > 0.0)
    {
        // what new average kinetic energy would be needed to maintain the outflow time-scale?
        const double radial_speed_ejec = 2.0*Gal[p].Rvir/Gal[p].OutflowTime - sqrt( 2.0*(Gal[p].OutflowSpecificEnergy - potential_and_thermal) );
        const double target_outflow_energy = thermal + Gal[p].EjectedPotential + 0.5 * sqr(radial_speed_ejec);
        
        if(!(radial_speed_ejec > 0.0 && target_outflow_energy>hot_specific_energy))
        {
            // can't preserve the time-scale for the outflow, assume all the outflowing gas is now fountaining
            if(Gal[p].FountainGas + Gal[p].OutflowGas > 0.0)
                Gal[p].FountainTime = (Gal[p].FountainGas * Gal[p].FountainTime + Gal[p].OutflowGas * t_dyn) / (Gal[p].FountainGas + Gal[p].OutflowGas);
            Gal[p].FountainGas += Gal[p].OutflowGas;
            Gal[p].MetalsFountainGas += Gal[p].MetalsOutflowGas;
            Gal[p].OutflowGas = 0.0;
            Gal[p].MetalsOutflowGas = 0.0;
            Gal[p].OutflowTime = 0.0;
        }
        else if(Gal[p].OutflowSpecificEnergy < target_outflow_energy)
        {
            // use energy conservation, assuming outflow timescale remains, to calculate previously outflowing gas that no longer has enough energy to make it outside Rvir
            reinc_mass = dmax(0.0, Gal[p].OutflowGas * (target_outflow_energy - Gal[p].OutflowSpecificEnergy)/target_outflow_energy);


            if(reinc_mass <= Gal[p].OutflowGas)
            {
                metallicity = get_metallicity(Gal[p].OutflowGas, Gal[p].MetalsOutflowGas);
                if(Gal[p].FountainGas + reinc_mass > 0.0)
                    Gal[p].FountainTime = (Gal[p].FountainGas * Gal[p].FountainTime + reinc_mass * t_dyn) / (Gal[p].FountainGas + reinc_mass);
                Gal[p].FountainGas += reinc_mass;
                Gal[p].MetalsFountainGas += (metallicity * reinc_mass);
                Gal[p].OutflowGas -= reinc_mass;
                Gal[p].MetalsOutflowGas -= (metallicity * reinc_mass);
                Gal[p].OutflowSpecificEnergy = target_outflow_energy; // this increases because it is the lower-energy material has been lost to the fountain reservoir
            }
            else
            {
                if(Gal[p].FountainGas + Gal[p].OutflowGas > 0.0)
                    Gal[p].FountainTime = (Gal[p].FountainGas * Gal[p].FountainTime + Gal[p].OutflowGas * t_dyn) / (Gal[p].FountainGas + Gal[p].OutflowGas);
                Gal[p].FountainGas += Gal[p].OutflowGas;
                Gal[p].MetalsFountainGas += Gal[p].MetalsOutflowGas;
                Gal[p].OutflowGas = 0.0;
                Gal[p].MetalsOutflowGas = 0.0;
                Gal[p].OutflowTime = 0.0;
            }
        }
    }
    
    if(Gal[p].EjectedMass > 0.0)
    {
        // what new average kinetic energy would be needed to maintain the reincorporation time-scale of the ejected reservoir?
        if(Gal[p].EjectedSpecificEnergy < Gal[p].EjectedPotential + thermal)
        { // ejected specific energy from last snapshot is no longer enough for the gas to be ejected at all
            if(Gal[p].FountainGas + Gal[p].EjectedMass > 0.0)
                Gal[p].FountainTime = (Gal[p].FountainGas * Gal[p].FountainTime + Gal[p].EjectedMass * t_dyn) / (Gal[p].FountainGas + Gal[p].EjectedMass);
            Gal[p].FountainGas += Gal[p].EjectedMass;
            Gal[p].MetalsFountainGas += Gal[p].MetalsEjectedMass;
            Gal[p].EjectedMass = 0.0;
            Gal[p].MetalsEjectedMass = 0.0;
            Gal[p].ReincTime = 0.0;
        }
        else
        {
            double R_min, R_max, R_guess, energy_guess, Rratio;
            int iter;
            const int iter_max=50;
            R_min = Gal[p].Rvir;
            R_max = 10 * Gal[p].Rvir;
            
            // ensure that the guess range for the turnaround radius is sufficiently large
            for(iter=0; iter<iter_max; iter++)
            {
                energy_guess = NFW_potential(p, R_max) + thermal;
                if(energy_guess < Gal[p].EjectedSpecificEnergy)
                {
                    R_min = R_max;
                    R_max *= 10.0;
                }
                else
                    break;
            }
            
            // iterate until finding the turnaround radius for gas ejected with the present specific energy
            for(iter=0; iter<iter_max; iter++)
            {
                R_guess = 0.5 * (R_min + R_max);
                energy_guess = NFW_potential(p, R_guess) + thermal;
                if(fabs((energy_guess-Gal[p].EjectedSpecificEnergy)/Gal[p].EjectedSpecificEnergy) < 1e-3)
                    break;
                
                if(energy_guess > Gal[p].EjectedSpecificEnergy)
                    R_max = R_guess;
                else
                    R_min = R_guess;
            }
            
            if(iter==iter_max) return;
            
            if(iter==iter_max)
            {
                printf("R_min, R_guess, R_max, Rvir = %e, %e, %e, %e\n", R_min, R_guess, R_max, Gal[p].Rvir);
                printf("energy_guess, Gal[p].EjectedSpecificEnergy = %e, %e\n", energy_guess, Gal[p].EjectedSpecificEnergy);
            }
            
            if(iter==iter_max) return;
            
            assert(iter<iter_max);
            Rratio = (R_guess-Gal[p].Rvir) / (R_guess-Gal[p].prevRvir);
            
            if(Rratio<1.0)
                Gal[p].ReincTime *= Rratio; // reduce reincorporation time based on the reduction in distance
            
            if(!(Gal[p].ReincTime >= 0.0 && Gal[p].ReincTime==Gal[p].ReincTime))
            {
                printf("Gal[p].ReincTime, Rratio = %e, %e\n", Gal[p].ReincTime, Rratio);
            }
            assert(Gal[p].ReincTime >= 0.0 && Gal[p].ReincTime==Gal[p].ReincTime);
        }

    }
    
}
