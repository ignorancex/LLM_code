#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "core_allvars.h"
#include "core_proto.h"



void starformation_and_feedback(int p, int centralgal, double dt, int step, double time, int k_now)
{
  double strdot, stars, reheated_mass, metallicity, stars_sum, area, SFE_H2, f_H2_const, DiscGasSum, DiscStarsSum, DiscPre, ColdPre;
  double r_inner, r_outer;
  double reff, tdyn, cold_crit, strdotfull, H2sum, new_metals; // For SFprescription==3

  double NewStars[N_BINS], NewStarsMetals[N_BINS];
  int i;
    
  double feedback_mass[4];
    
  // these terms only used when SupernovaRecipeOn>=3
  double hot_specific_energy, hot_thermal_and_kinetic, j_hot, ejected_cold_mass;
    
  j_hot = 2 * Gal[p].Vvir * Gal[p].CoolScaleRadius;
  hot_thermal_and_kinetic = 0.5 * (sqr(Gal[p].Vvir) + sqr(j_hot)/Gal[p].R2_hot_av);
  hot_specific_energy = Gal[p].HotGasPotential + hot_thermal_and_kinetic;  

  double StarsPre = Gal[p].StellarMass;
  check_channel_stars(p);

  reheated_mass = 0.0; // initialise
    double reheated_sum = 0.0;
    
  // Checks that the deconstructed disc is being treated properly and not generating NaNs
  DiscGasSum = get_disc_gas(p);
  DiscStarsSum = get_disc_stars(p);
  assert(DiscGasSum <= 1.01*Gal[p].ColdGas && DiscGasSum >= Gal[p].ColdGas/1.01);
  assert(DiscStarsSum <= 1.01*Gal[p].StellarMass);
  assert(Gal[p].HotGas == Gal[p].HotGas && Gal[p].HotGas >= 0);
  assert(Gal[centralgal].HotGas >= Gal[centralgal].MetalsHotGas);

  f_H2_const = 1.38e-3 * pow((CM_PER_MPC*CM_PER_MPC/1e12 / SOLAR_MASS) * (UnitMass_in_g / UnitLength_in_cm / UnitLength_in_cm), 2.0*H2FractionExponent);
  SFE_H2 = 4.35e-4 / Hubble_h * UnitTime_in_s / SEC_PER_MEGAYEAR; // This says if SfrEfficiency==1.0, then the time-scale for H2 consumption is 2.3 Gyr (Bigiel et al. 2011)

  // Initialise variables
  strdot = 0.0;
  strdotfull = 0.0;
  stars_sum = 0.0;

  Gal[p].SfrDiskColdGas[step] = Gal[p].ColdGas;
  Gal[p].SfrDiskColdGasMetals[step] = Gal[p].MetalsColdGas; // I believe TAO wants these fields.  Otherwise irrelevant
    
  update_HI_H2(p, time, k_now);
    
  if(SFprescription==2) // Prescription based on SAGE
  {
      reff = 3.0 * Gal[p].DiskScaleRadius;
      tdyn = reff / Gal[p].Vvir;
      cold_crit = 0.19 * Gal[p].Vvir * reff;
      if(Gal[p].ColdGas > cold_crit && tdyn > 0.0)
          strdotfull = SfrEfficiency * (Gal[p].ColdGas - cold_crit) / tdyn;
      
      H2sum = 0.0;
      for(i=N_BINS-1; i>=0; i--) H2sum += Gal[p].DiscH2[i];
  }

  for(i=0; i<N_BINS; i++)
  {
	if(Gal[p].DiscStarsMetals[i] > Gal[p].DiscStars[i]) printf("DiscStars, Metals = %e, %e\n", Gal[p].DiscStars[i], Gal[p].DiscStarsMetals[i]);
	assert(Gal[p].DiscStarsMetals[i] <= Gal[p].DiscStars[i]);

    r_inner = Gal[p].DiscRadii[i];
    r_outer = Gal[p].DiscRadii[i+1];
      
    area = M_PI * (r_outer*r_outer - r_inner*r_inner);
    if(Gal[p].Vvir>0)
    {
      if(SFprescription==1 && Gal[p].DiscH2[i]<0.5*Gal[p].DiscHI[i])
      {
          double bb = sqrt(Gal[p].DiscStars[i]*Gal[p].DiscStars[0])/area; // quadratic b term
          double cc = -pow(0.5/f_H2_const, 1.0/0.92);
          double Sig_gas_half = 0.5*(-bb + sqrt(bb*bb-4.0*cc));
          double SFE_gas = SFE_H2 * 0.75 * 1.0/(1.0/0.5 + 1) * (1 - Gal[p].DiscGasMetals[i]/Gal[p].DiscGas[i])/1.3 / Sig_gas_half;
          strdot = SfrEfficiency * SFE_gas * sqr(Gal[p].DiscGas[i]) / area;
      }
      else if(SFprescription==2)
      {
          if(Gal[p].ColdGas>0.0)
              strdot = strdotfull * Gal[p].DiscGas[i] / Gal[p].ColdGas;// * Gal[p].DiscH2[i] / H2sum;
          else
              strdot = 0.0;
      }
      else
          strdot = SfrEfficiency * SFE_H2 * Gal[p].DiscH2[i];
    }
    else // These galaxies (which aren't useful for science) won't have H2 to form stars
      strdot = 0.0;

    stars = strdot * dt;

    if(stars < MIN_STARFORMATION)
        stars = 0.0;

    if(stars > Gal[p].DiscGas[i])
        stars = Gal[p].DiscGas[i];
      
    calculate_feedback_masses(p, stars, i, Gal[p].DiscGas[i], hot_specific_energy, feedback_mass);
    reheated_mass = feedback_mass[0];
    stars = feedback_mass[2];
    ejected_cold_mass = feedback_mass[3];

    Gal[p].StellarFormationMassAge[k_now] += stars;
    Gal[p].DiscSFR[i] += stars / dt;
	stars_sum += stars;
    reheated_sum += reheated_mass;

	DiscPre = Gal[p].DiscGas[i];
	ColdPre = Gal[p].ColdGas;

    // Update for star formation
    metallicity = get_metallicity(Gal[p].DiscGas[i], Gal[p].DiscGasMetals[i]);
	assert(Gal[p].DiscGasMetals[i] <= Gal[p].DiscGas[i]);
    if(stars>=MIN_STARS_FOR_SN)
    {
        NewStars[i] = (1 - RecycleFraction) * stars;
        NewStarsMetals[i] = (1 - RecycleFraction) * metallicity * stars;
    }
    else
    {
        NewStars[i] = stars;
        NewStarsMetals[i] = metallicity * stars;
    }
    if(!(NewStarsMetals[i] <= NewStars[i]))
    {
      printf("NewStars, metals = %e, %e\n", NewStars[i], NewStarsMetals[i]);
      printf("Gas, metals = %e, %e\n", Gal[p].DiscGas[i], Gal[p].DiscGasMetals[i]);
    }
	assert(NewStarsMetals[i] <= NewStars[i]);
    update_from_star_formation(p, stars, metallicity, i);

    // These checks ensure numerical uncertainties don't blow up    
	if(Gal[p].DiscStarsMetals[i] > Gal[p].DiscStars[i]) printf("DiscStars, Metals = %e, %e\n", Gal[p].DiscStars[i], Gal[p].DiscStarsMetals[i]);
	assert(Gal[p].DiscStarsMetals[i] <= Gal[p].DiscStars[i]);

	if(reheated_mass > Gal[p].DiscGas[i] && reheated_mass < 1.01*Gal[p].DiscGas[i])
	  reheated_mass = Gal[p].DiscGas[i];

    assert(fabs(Gal[p].ColdGas-ColdPre) <= 1.01*fabs(Gal[p].DiscGas[i]-DiscPre) && fabs(Gal[p].ColdGas-ColdPre) >= 0.999*fabs(Gal[p].DiscGas[i]-DiscPre) && (Gal[p].ColdGas-ColdPre)*(Gal[p].DiscGas[i]-DiscPre)>=0.0);
 
	DiscPre = Gal[p].DiscGas[i];
	ColdPre = Gal[p].ColdGas;

	// Inject new metals from SN
	if(SupernovaRecipeOn > 0 && stars>=MIN_STARS_FOR_SN)
	{
        new_metals = Yield * stars*(1-metallicity) * RecycleFraction/FinalRecycleFraction;
        Gal[p].DiscGasMetals[i] += new_metals;
        Gal[p].MetalsColdGas += new_metals;	
    }
    if(Gal[p].DiscGasMetals[i] > Gal[p].DiscGas[i]) printf("DiscGas, Metals = %e, %e\n", Gal[p].DiscGas[i], Gal[p].DiscGasMetals[i]);
	assert(Gal[p].DiscGasMetals[i]<=Gal[p].DiscGas[i]);
	if(Gal[p].DiscStarsMetals[i] > Gal[p].DiscStars[i]) printf("DiscStars, Metals = %e, %e\n", Gal[p].DiscStars[i], Gal[p].DiscStarsMetals[i]);
	assert(Gal[p].DiscStarsMetals[i] <= Gal[p].DiscStars[i]);

      // Update from SN feedback
      metallicity = get_metallicity(Gal[p].DiscGas[i], Gal[p].DiscGasMetals[i]);
      assert(Gal[p].DiscGasMetals[i] <= Gal[p].DiscGas[i]);
      assert(reheated_mass==reheated_mass && reheated_mass!=INFINITY);
      assert(Gal[p].MetalsHotGas>=0);
      assert(Gal[p].MetalsHotGas<=Gal[p].HotGas);
      if(reheated_mass + ejected_cold_mass > 0.0)
      {
          assert(Gal[p].OutflowGas >= 0.0);
        update_from_feedback(p, reheated_mass, metallicity, i, ejected_cold_mass);
      }

      if(!(fabs(Gal[p].ColdGas-ColdPre) <= 1.01*fabs(Gal[p].DiscGas[i]-DiscPre) && fabs(Gal[p].ColdGas-ColdPre) >= 0.999*fabs(Gal[p].DiscGas[i]-DiscPre) && (Gal[p].ColdGas-ColdPre)*(Gal[p].DiscGas[i]-DiscPre)>=0.0))
          printf("fabs(Gal[p].ColdGas-ColdPre), fabs(Gal[p].DiscGas[i]-DiscPre) = %e, %e\n", fabs(Gal[p].ColdGas-ColdPre), fabs(Gal[p].DiscGas[i]-DiscPre));
        assert(fabs(Gal[p].ColdGas-ColdPre) <= 1.01*fabs(Gal[p].DiscGas[i]-DiscPre) && fabs(Gal[p].ColdGas-ColdPre) >= 0.999*fabs(Gal[p].DiscGas[i]-DiscPre) && (Gal[p].ColdGas-ColdPre)*(Gal[p].DiscGas[i]-DiscPre)>=0.0);


  }
    
  double NewStarSum = 0.0;
  for(i=N_BINS-1; i>=0; i--) NewStarSum += NewStars[i];
    
  // Sum stellar discs together
  if(NewStarSum>0.0)
    combine_stellar_discs(p, NewStars, NewStarsMetals, time);

  // Update the star formation rate 
  Gal[p].SfrFromH2[step] += stars_sum / dt;
  Gal[p].StarsFromH2 += NewStarSum;
    
  if(Gal[p].StellarMass >= MIN_STARS_FOR_SN)
    {
      check_channel_stars(p);
      assert(Gal[p].StellarMass >= (StarsPre + NewStarSum)/1.01 && Gal[p].StellarMass <= (StarsPre + NewStarSum)*1.01);
  }


  DiscGasSum = get_disc_gas(p);
  assert(DiscGasSum <= 1.01*Gal[p].ColdGas && DiscGasSum >= Gal[p].ColdGas*0.99);
  assert(Gal[centralgal].HotGas >= Gal[centralgal].MetalsHotGas);

}


void calculate_feedback_masses(int p, double stars, int i, double max_consume, double hot_specific_energy, double *feedback_mass)
{
    // principle function to calculate how much mass is reheated and ejected from stellar feedback after a period of star formation (also see delayed_feedback for older generations of stars)
    
    double reheated_mass, ejected_cold_mass, fac;
    double energy_feedback, annulus_radius, annulus_velocity, cold_specific_energy, reheat_specific_energy, escape_velocity2, vertical_velocity, v_therm2;
    double v_wind, v_wind2, new_ejected_specific_energy=0.0;
    
    if(max_consume > Gal[p].DiscGas[i])
        max_consume = Gal[p].DiscGas[i];
    
    if(stars > max_consume)
        stars = max_consume;
        
    if(SupernovaRecipeOn > 0 && Gal[p].DiscGas[i] > 0.0 && stars>=MIN_STARS_FOR_SN)
    {
        energy_feedback = stars * EnergySNcode * SNperMassFormed; // still controlled by a coupling efficiency for practical testing purposes

        annulus_radius = sqrt(0.5 * (sqr(Gal[p].DiscRadii[i]) + sqr(Gal[p].DiscRadii[i+1])) );
        annulus_velocity = 0.5 * (DiscBinEdge[i] + DiscBinEdge[i+1]) / annulus_radius;
        vertical_velocity = (1.1e6 + 1.13e6 * ZZ[Gal[p].SnapNum])/UnitVelocity_in_cm_per_s;
        cold_specific_energy = 0.5*(sqr(annulus_velocity) + sqr(vertical_velocity) + Gal[p].Potential[i] + Gal[p].Potential[i+1]);
        reheat_specific_energy = hot_specific_energy - cold_specific_energy;

        v_therm2 = sqr(Gal[p].Vvir);
        v_wind2 = 2.0*reheat_specific_energy - v_therm2;
        v_wind = sqrt(v_wind2);
        escape_velocity2 = 2.0 * (Gal[p].EjectedPotential - Gal[p].Potential[i]);
        
        if(reheat_specific_energy < 0.0) // possible to happen.  Means ejecting is the only logical choice but can't trust v_wind now
        {
            new_ejected_specific_energy = dmax(Gal[p].OutflowSpecificEnergy, dmax(Gal[p].EjectedPotential, cold_specific_energy) + 0.5*v_therm2);
            ejected_cold_mass = energy_feedback / (new_ejected_specific_energy - cold_specific_energy);
            reheated_mass = 0.0;
        }
        else if(sqr(v_wind - annulus_velocity) > escape_velocity2) // the wind speed is high enough to eject all the gas instead of just reheating it
        {
            reheated_mass = 0.0;
            new_ejected_specific_energy = dmax(cold_specific_energy + 0.5*(v_therm2 + v_wind2), Gal[p].EjectedPotential + 0.5*v_therm2);
            ejected_cold_mass = energy_feedback / (new_ejected_specific_energy - cold_specific_energy);
        }
        else if(sqr(v_wind + annulus_velocity) > escape_velocity2) // the wind speed is enough to eject some of the gas
        {
            double reheated_ejected_ratio = 1.0 / (0.5 - (escape_velocity2 - sqr(annulus_velocity) - sqr(v_wind)) / (4.0 * annulus_velocity * v_wind) ) - 1.0;
            new_ejected_specific_energy = 0.25*(Gal[p].Potential[i] + Gal[p].Potential[i+1] + sqr(v_wind + annulus_velocity)) + 0.5*(Gal[p].EjectedPotential + v_therm2);
            ejected_cold_mass = energy_feedback / (reheated_ejected_ratio * reheat_specific_energy + new_ejected_specific_energy - cold_specific_energy);
            reheated_mass = reheated_ejected_ratio * ejected_cold_mass;
        }
        else
        {
            reheated_mass = energy_feedback / reheat_specific_energy;
            ejected_cold_mass = 0.0;
            new_ejected_specific_energy = 0.0;
        }
   
        if(!(reheated_mass>=0) || !(reheated_mass>=0))
        {
            printf("reheated_mass, reheat_specific_energy = %e, %e\n", reheated_mass, reheat_specific_energy);
            printf("ejected_cold_mass = %e\n", ejected_cold_mass);
        }
            
        assert(ejected_cold_mass>=0);
        assert(reheated_mass>=0);
            
        
        // Can't use more cold gas than is available, so balance SF and feedback 
        if(((1-RecycleFraction)*stars + reheated_mass + ejected_cold_mass)>max_consume && (stars + reheated_mass + ejected_cold_mass)>0.0)
        {
                fac = max_consume / ((1-RecycleFraction) * stars + reheated_mass + ejected_cold_mass);
                stars *= fac;
                reheated_mass *= fac;
                energy_feedback *= fac;
                ejected_cold_mass *= fac;
        }
        

        if(stars<MIN_STARS_FOR_SN)
        {
          if(max_consume >= MIN_STARS_FOR_SN)
          {
              stars = MIN_STARS_FOR_SN;
              reheated_mass = max_consume - stars; // Used to have (1-RecycleFraction)* in front of stars here, but changed philosophy
              assert(reheated_mass==reheated_mass && reheated_mass!=INFINITY);
          }
          else
          {
              stars = max_consume;
              reheated_mass = 0.0;
          }
          ejected_cold_mass = 0.0;
        }
        
    }
    else // I haven't actually dealt with the situation of Supernovae being turned off here.  But do I even want to turn SN off?
    {
      reheated_mass = 0.0;
      ejected_cold_mass = 0.0;
    }

    // update the specific energy of the outflowing reservoir (where the stuff to be ejected goes first)
    if(ejected_cold_mass > 0.0)
    {
        if(!(new_ejected_specific_energy - Gal[p].Potential[0] - 0.5*sqr(Gal[p].Vvir) > 0.0))
        {
            printf("new_ejected_specific_energy, Gal[p].Potential[0], 0.5*sqr(Gal[p].Vvir) = %e, %e, %e\n", new_ejected_specific_energy, Gal[p].Potential[0], 0.5*sqr(Gal[p].Vvir));
            printf("reheated_mass, ejected_cold_mass = %e, %e\n", reheated_mass, ejected_cold_mass);
            printf("v_wind = %e\n", v_wind);
            printf("sqr(annulus_velocity + v_wind), escape_velocity2, sqr(annulus_velocity - v_wind) = %e, %e, %e\n", sqr(annulus_velocity + v_wind), escape_velocity2, sqr(annulus_velocity - v_wind));
        }
        
        
        assert(new_ejected_specific_energy - Gal[p].Potential[0] - 0.5*sqr(Gal[p].Vvir) > 0.0);
        
        assert(Gal[p].OutflowGas + ejected_cold_mass >= 0.0);
        Gal[p].OutflowSpecificEnergy = (Gal[p].OutflowGas * Gal[p].OutflowSpecificEnergy + ejected_cold_mass * new_ejected_specific_energy) / (Gal[p].OutflowGas + ejected_cold_mass);
        update_outflow_time(p, ejected_cold_mass, new_ejected_specific_energy);
    }
        
    feedback_mass[0] = reheated_mass;
    feedback_mass[2] = stars;
    feedback_mass[3] = ejected_cold_mass;
        
}


void update_from_star_formation(int p, double stars, double metallicity, int i)
{
  // In older SAGE, this updated the gas and stellar components.  It only does the gas component now due to the way in which discs are handled.

  // update gas and metals from star formation
  if(stars>=MIN_STARS_FOR_SN)
  {
      Gal[p].DiscGas[i] -= (1 - RecycleFraction) * stars;
      Gal[p].DiscGasMetals[i] -= metallicity * (1 - RecycleFraction) * stars;

      if(Gal[p].DiscGasMetals[i] > Gal[p].DiscGas[i])
        printf("update_from_star_formation report -- gas metals, gas mass = %e, %e\n", Gal[p].DiscGasMetals[i], Gal[p].DiscGas[i]);
      
      Gal[p].ColdGas -= (1 - RecycleFraction) * stars;
      Gal[p].MetalsColdGas -= metallicity * (1 - RecycleFraction) * stars;
  }
  else
  {
      Gal[p].DiscGas[i] -= stars;
      Gal[p].DiscGasMetals[i] -= metallicity * stars;
      Gal[p].ColdGas -= stars;
      Gal[p].MetalsColdGas -= metallicity * stars;
  }
    
    
  if(Gal[p].DiscGas[i] <= 0.0){
	Gal[p].DiscGas[i]=0.0;
	Gal[p].DiscGasMetals[i]=0.0;}

  if(Gal[p].ColdGas <= 0.0){
	Gal[p].ColdGas=0.0;
	Gal[p].MetalsColdGas=0.0;}
  
}



void update_from_feedback(int p, double reheated_mass, double metallicity, int i, double ejected_cold_mass)
{
  assert(Gal[p].MetalsHotGas <= Gal[p].HotGas);
  assert(Gal[p].HotGas>=0);
  assert(Gal[p].MetalsHotGas>=0);
  assert(metallicity>=0);
  assert(reheated_mass>=0);
  assert(ejected_cold_mass>=0);
  assert(Gal[p].OutflowGas >= 0.0);
  
  double reheat_eject_sum = reheated_mass + ejected_cold_mass;

  if(SupernovaRecipeOn>0 && reheat_eject_sum>MIN_STARFORMATION) // Imposing a minimum. Chosen as MIN_STARFORMATION for convenience.  Makes sense to be same order of magnitude though.
  {
      
      
    if(reheat_eject_sum < Gal[p].DiscGas[i])
    {
        Gal[p].DiscGas[i] -= reheat_eject_sum;
        Gal[p].ColdGas -= reheat_eject_sum;
        Gal[p].EjectedSNGasMass += reheat_eject_sum;
    }
    else
    {
        reheat_eject_sum = Gal[p].DiscGas[i]; // this is just about precision accuracy, nothing more
        Gal[p].EjectedSNGasMass += Gal[p].DiscGas[i];
        Gal[p].ColdGas -= Gal[p].DiscGas[i];
        Gal[p].DiscGas[i] = 0.0;
    }

    if(reheated_mass > 0.0)
        Gal[p].FountainTime = (Gal[p].FountainGas * Gal[p].FountainTime + reheated_mass * 0.1 / sqrt(Hubble_sqr_z(Halo[Gal[p].HaloNr].SnapNum))) / (Gal[p].FountainGas + reheated_mass);
    Gal[p].FountainGas += reheated_mass;
    Gal[p].OutflowGas += ejected_cold_mass;
      assert(Gal[p].OutflowGas >= 0.0);

	if(Gal[p].DiscGas[i]>0.0)
	{
        
      metallicity = dmax(0.1*metallicity, (Gal[p].DiscGasMetals[i] - Gal[p].DiscGas[i])/reheat_eject_sum ); // assume only 10% of metals entrained in an outflow, but ensure metals never exceed total mass of gas left behind. 
        
      // Factors of 1.0001 and 0.9999 is to avoid numerical issues with metals reading greater than total gas.
	  Gal[p].MetalsColdGas -= 1.0001 * metallicity * reheat_eject_sum;
      Gal[p].DiscGasMetals[i] -= 1.0001 *metallicity * reheat_eject_sum;

      Gal[p].MetalsFountainGas += 0.9999 * metallicity * reheated_mass;
      Gal[p].MetalsOutflowGas += 0.9999 * metallicity * ejected_cold_mass;

        if(!(Gal[p].DiscGasMetals[i] <= Gal[p].DiscGas[i]))
        {
            printf("Gal[p].DiscGasMetals[i], Gal[p].DiscGas[i], metallicity, reheat_eject_sum = %e, %e, %e, %e\n", Gal[p].DiscGasMetals[i], Gal[p].DiscGas[i], metallicity, reheat_eject_sum);
        }

        
        if(Gal[p].OutflowGas <= 0.0)
        {
            Gal[p].OutflowGas = 0.0;
            Gal[p].MetalsOutflowGas = 0.0;
        }
        
        if(Gal[p].FountainGas <= 0.0)
        {
            Gal[p].FountainGas = 0.0;
            Gal[p].MetalsFountainGas = 0.0;
        }
        
        if(!(Gal[p].MetalsOutflowGas <= Gal[p].OutflowGas))
        {
            printf("Gal[p].MetalsOutflowGas, Gal[p].OutflowGas, metallicity, ejected_cold_mass = %e, %e, %e, %e\n", Gal[p].MetalsOutflowGas, Gal[p].OutflowGas, metallicity, ejected_cold_mass);
        }
        
      assert(Gal[p].MetalsOutflowGas <= Gal[p].OutflowGas);
        
        assert(Gal[p].DiscGasMetals[i] <= Gal[p].DiscGas[i]);
      assert(Gal[p].MetalsFountainGas <= Gal[p].FountainGas);


	}
	else
	{

        Gal[p].MetalsFountainGas += Gal[p].DiscGasMetals[i] * reheated_mass/reheat_eject_sum;
        Gal[p].MetalsOutflowGas += Gal[p].DiscGasMetals[i] * ejected_cold_mass/reheat_eject_sum;

        
      Gal[p].MetalsColdGas -= Gal[p].DiscGasMetals[i];
      Gal[p].DiscGasMetals[i] = 0.0;
    }
  }


  if(Gal[p].DiscGas[i] <= 0.0)
  {
	Gal[p].DiscGas[i]=0.0;
	Gal[p].DiscGasMetals[i]=0.0;
  }

  if(Gal[p].ColdGas <= 0.0)
  {
	Gal[p].ColdGas=0.0;
	Gal[p].MetalsColdGas=0.0;
  }

  if(Gal[p].HotGas <= 0.0)
  {
	Gal[p].HotGas=0.0;
	Gal[p].MetalsHotGas=0.0;
  }
    
    assert(reheated_mass>=0);
    assert(ejected_cold_mass>=0);
  Gal[p].SNreheatRate += reheated_mass;
  Gal[p].SNejectRate += ejected_cold_mass;
    
    assert(Gal[p].HotGas>=0);
    assert(Gal[p].MetalsHotGas>=0);

}


void update_from_ejection(int p, double ejected_mass)
{    // feedback from spheroidal stars and excess energy from quasars will cause CGM gas to outflow    
    double metallicityHot = get_metallicity(Gal[p].HotGas, Gal[p].MetalsHotGas);
    double metallicityFountain = get_metallicity(Gal[p].FountainGas, Gal[p].MetalsFountainGas);
    double hot_fraction = Gal[p].HotGas / (Gal[p].HotGas + Gal[p].FountainGas);
    
    assert(ejected_mass >= 0.0);
    
    if(Gal[p].OutflowSpecificEnergy < Gal[p].EjectedPotential + 0.5*sqr(Gal[p].Vvir))
        Gal[p].OutflowSpecificEnergy = Gal[p].EjectedPotential + 0.5*sqr(Gal[p].Vvir);
    
    if(!(Gal[p].OutflowSpecificEnergy > Gal[p].Potential[0] + 0.5*sqr(Gal[p].Vvir)))
    {
        printf("Gal[p].OutflowSpecificEnergy = %e\n", Gal[p].OutflowSpecificEnergy);
        printf("0.5*sqr(Gal[p].Vvir) = %e\n", 0.5*sqr(Gal[p].Vvir));
        printf("Gal[p].Potential[0], Gal[p].EjectedPotential = %e, %e\n", Gal[p].Potential[0], Gal[p].EjectedPotential);
    }
    
    assert(Gal[p].OutflowSpecificEnergy > Gal[p].Potential[0] + 0.5*sqr(Gal[p].Vvir));

    // strictly speaking, shouldn't the distance and initial speeds be different compared to gas that goes from cold->outflowing directly?
    if(ejected_mass >= Gal[p].HotGas + Gal[p].FountainGas)
    {
        update_outflow_time(p, Gal[p].FountainGas, Gal[p].OutflowSpecificEnergy);
        Gal[p].OutflowGas += (Gal[p].HotGas + Gal[p].FountainGas);
        assert(Gal[p].OutflowGas >= 0.0);
        Gal[p].MetalsOutflowGas += (Gal[p].MetalsHotGas + Gal[p].MetalsFountainGas);
        Gal[p].HotGas = 0.0;
        Gal[p].MetalsHotGas = 0.0;
        Gal[p].FountainGas = 0.0;
        Gal[p].MetalsFountainGas = 0.0;
    }
    else
    {
        update_outflow_time(p, ejected_mass, Gal[p].OutflowSpecificEnergy);
        Gal[p].OutflowGas += ejected_mass;
        assert(Gal[p].OutflowGas >= 0.0);

        Gal[p].MetalsOutflowGas += (ejected_mass * (hot_fraction * metallicityHot + (1.0-hot_fraction) * metallicityFountain));
        Gal[p].HotGas -= (ejected_mass * hot_fraction);
        Gal[p].MetalsHotGas -= (ejected_mass * hot_fraction * metallicityHot);
        Gal[p].FountainGas -= (ejected_mass * (1.0-hot_fraction));
        Gal[p].MetalsFountainGas -= (ejected_mass * (1.0-hot_fraction) * metallicityFountain);
    }
        
}


void combine_stellar_discs(int p, double NewStars[N_BINS], double NewStarsMetals[N_BINS], double time)
{
	double sdisc_spin_mag, J_sdisc, J_new, J_retro, J_sum, cos_angle_sdisc_comb, cos_angle_new_comb, DiscStarSum;
	double SDiscNewSpin[3];
	double Disc1[N_BINS], Disc1Metals[N_BINS], Disc2[N_BINS], Disc2Metals[N_BINS];
//    double Disc1Age[N_AGE_BINS][N_BINS], Disc1MetalsAge[N_AGE_BINS][N_BINS]; // NOTE THE REVERSE ORDER HERE!
    double Disc1Age[N_BINS][N_AGE_BINS], Disc1MetalsAge[N_BINS][N_AGE_BINS]; 
    double Disc1VelDisp[N_BINS], Disc1VelDispAge[N_BINS][N_AGE_BINS];
	int i, k, k_now;
    
    double sigma_gas = (1.1e6 + 1.13e6 * ZZ[Gal[p].SnapNum])/UnitVelocity_in_cm_per_s;
	
    // Determine which age bin new stars should be put into
    k_now = get_stellar_age_bin_index(time);
    
    DiscStarSum = get_disc_stars(p);
    
	for(i=0; i<N_BINS; i++)
    {
		assert(Gal[p].DiscStarsMetals[i] <= Gal[p].DiscStars[i]);
		assert(Gal[p].DiscGasMetals[i] <= Gal[p].DiscGas[i]);
		assert(NewStarsMetals[i] <= NewStars[i]);
        
        for(k=k_now; k<N_AGE_BINS; k++)
            assert(Gal[p].DiscStarsMetalsAge[i][k] <= Gal[p].DiscStarsAge[i][k]);
    }
	
	// Try not to get confused, where "new" here implies the newly formed stars.  In the cooling recipe, "new" meant the combined disc, here instead denoted "comb".
	
	J_new = 0.0;
	for(i=N_BINS-1; i>=0; i--)
		J_new += NewStars[i] * (DiscBinEdge[i]+DiscBinEdge[i+1])/2.0; // The assumption that DiscBinEdge is now proportional to radius has broken down
	
	// Determine projection angles for combining discs
	if(Gal[p].StellarMass > Gal[p].SecularBulgeMass + Gal[p].ClassicalBulgeMass)
	{
		// Ensure the stellar disc spin magnitude is normalised
		sdisc_spin_mag = sqrt(sqr(Gal[p].SpinStars[0]) + sqr(Gal[p].SpinStars[1]) + sqr(Gal[p].SpinStars[2]));
		assert(sdisc_spin_mag==sdisc_spin_mag);
		if(sdisc_spin_mag>0.0)
		{
			for(i=0; i<3; i++)
			{
				Gal[p].SpinStars[i] /= sdisc_spin_mag; 
                assert(Gal[p].SpinStars[i]==Gal[p].SpinStars[i]);
			}
		}
	
		J_sdisc = get_disc_ang_mom(p, 1);
	
		// Obtain new spin vector of stellar disc
        for(i=0; i<3; i++){
			SDiscNewSpin[i] = Gal[p].SpinGas[i]*J_new + Gal[p].SpinStars[i]*J_sdisc;
            assert(SDiscNewSpin[i]==SDiscNewSpin[i]);}
        
		// Normalise the new spin
		sdisc_spin_mag = sqrt(sqr(SDiscNewSpin[0]) + sqr(SDiscNewSpin[1]) + sqr(SDiscNewSpin[2]));
        if(sdisc_spin_mag>0.0)
        {
            for(i=0; i<3; i++)
                SDiscNewSpin[i] /= sdisc_spin_mag;
        }
		
        sdisc_spin_mag = sqrt(sqr(SDiscNewSpin[0]) + sqr(SDiscNewSpin[1]) + sqr(SDiscNewSpin[2]));
        if(sdisc_spin_mag<0.99 || sdisc_spin_mag>1.01)
        {
            printf("SpinStars somehow became %e\n", sdisc_spin_mag);
            printf("with J_sdisc, J_new = %e, %e\n", J_sdisc, J_new);
            printf("DiscStars, DiscGas = %e, %e\n", get_disc_stars(p), get_disc_gas(p));
        }
        assert(sdisc_spin_mag >= 0.99 && sdisc_spin_mag <= 1.01);
        
		cos_angle_sdisc_comb = Gal[p].SpinStars[0]*SDiscNewSpin[0] + Gal[p].SpinStars[1]*SDiscNewSpin[1] + Gal[p].SpinStars[2]*SDiscNewSpin[2];
		cos_angle_new_comb = Gal[p].SpinGas[0]*SDiscNewSpin[0] + Gal[p].SpinGas[1]*SDiscNewSpin[1] + Gal[p].SpinGas[2]*SDiscNewSpin[2];
	}
	else
	{
		cos_angle_sdisc_comb = 1.0;
		cos_angle_new_comb = 1.0;
		J_sdisc = 0.0;
        for(i=0; i<3; i++)
            SDiscNewSpin[i] = Gal[p].SpinGas[i];
	}
	
	// Combine the discs
	if(cos_angle_sdisc_comb<1.0)
    {
		project_disc_with_dispersion(Gal[p].DiscStars, Gal[p].DiscStarsMetals, Gal[p].VelDispStars, Gal[p].DiscStarsAge, Gal[p].DiscStarsMetalsAge, Gal[p].VelDispStarsAge, cos_angle_sdisc_comb, p, k_now, Disc1, Disc1Metals, Disc1VelDisp, Disc1Age, Disc1MetalsAge, Disc1VelDispAge);
        
        // using old functions here as I don't need to project dispersion
		project_disc(NewStars, cos_angle_new_comb, p, Disc2);
		project_disc(NewStarsMetals, cos_angle_new_comb, p, Disc2Metals);
		
        Gal[p].StellarMass = Gal[p].SecularBulgeMass + Gal[p].ClassicalBulgeMass;
        Gal[p].MetalsStellarMass = Gal[p].SecularMetalsBulgeMass + Gal[p].ClassicalMetalsBulgeMass;
		for(i=N_BINS-1; i>=0; i--)
		{
            if(Disc1[i]+Disc2[i] <= 0) continue;

            if(Gal[p].DiscStarsMetals[i] > Gal[p].DiscStars[i]) printf("DiscStars, Metals = %e, %e\n", Gal[p].DiscStars[i], Gal[p].DiscStarsMetals[i]);
			assert(Gal[p].DiscStarsMetals[i] <= Gal[p].DiscStars[i]);
            
            Gal[p].VelDispStars[i] = sqrt( (Disc1[i]*sqr(Disc1VelDisp[i]) + Disc2[i]*sqr(sigma_gas)) / (Disc1[i]+Disc2[i]) );
            assert(Gal[p].VelDispStars[i] >= 0);
			Gal[p].DiscStars[i] = Disc1[i] + Disc2[i];
			Gal[p].DiscStarsMetals[i] = Disc1Metals[i] + Disc2Metals[i];
            
			if(Gal[p].DiscStars[i]==0.0 && Gal[p].DiscStarsMetals[i] < 1e-20) Gal[p].DiscStarsMetals[i] = 0.0;
            Gal[p].StellarMass += Gal[p].DiscStars[i];
            Gal[p].MetalsStellarMass += Gal[p].DiscStarsMetals[i];
            
			if(Gal[p].DiscStarsMetals[i] > Gal[p].DiscStars[i]) printf("DiscStars, Metals = %e, %e\n", Gal[p].DiscStars[i], Gal[p].DiscStarsMetals[i]);
			assert(Gal[p].DiscStarsMetals[i] <= Gal[p].DiscStars[i]);
            
            if(AgeStructOut>0)
            {
                for(k=k_now; k<N_AGE_BINS; k++)
                {
                    Gal[p].DiscStarsAge[i][k] = 1.0*Disc1Age[i][k];
                    Gal[p].DiscStarsMetalsAge[i][k] = 1.0*Disc1MetalsAge[i][k];
                    Gal[p].VelDispStarsAge[i][k] = 1.0*Disc1VelDispAge[i][k];
                    assert(Disc1VelDispAge[i][k] >= 0);
                    assert(Gal[p].VelDispStarsAge[i][k] >= 0);
                    
                    assert(Gal[p].DiscStarsMetalsAge[i][k] <= Gal[p].DiscStarsAge[i][k]);
                }

                
                if(Disc1Age[i][k_now] + Disc2[i] > 0)
                {
                    Gal[p].DiscStarsAge[i][k_now] += 1.0*Disc2[i];
                    Gal[p].DiscStarsMetalsAge[i][k_now] += 1.0*Disc2Metals[i];
                    Gal[p].VelDispStarsAge[i][k_now] = sqrt( (Disc1Age[i][k_now] * sqr(Disc1VelDispAge[i][k_now]) + Disc2[i] * sqr(sigma_gas)) / (Disc1Age[i][k_now] + Disc2[i]) );
                    assert(Gal[p].VelDispStarsAge[i][k_now] >= 0);
                }
            }
		}
	}
    else
	{
        DiscStarSum = get_disc_stars(p);
        double NewStarSum = 0.0;
        for(i=N_BINS-1; i>=0; i--) NewStarSum += NewStars[i];
		for(i=N_BINS-1; i>=0; i--)
		{
            if(Gal[p].DiscStars[i] + NewStars[i] <= 0) continue;
            
			if(Gal[p].DiscStarsMetals[i] > Gal[p].DiscStars[i]) printf("DiscStars, Metals = %e, %e\n", Gal[p].DiscStars[i], Gal[p].DiscStarsMetals[i]);
			assert(Gal[p].DiscStarsMetals[i] <= Gal[p].DiscStars[i]);
            
            Gal[p].VelDispStars[i] = sqrt( (Gal[p].DiscStars[i]*sqr(Gal[p].VelDispStars[i]) + NewStars[i]*sqr(sigma_gas)) / (Gal[p].DiscStars[i] + NewStars[i])  );
            assert(Gal[p].VelDispStars[i] >= 0);
			Gal[p].DiscStars[i] += NewStars[i];
			Gal[p].DiscStarsMetals[i] += NewStarsMetals[i];
            Gal[p].StellarMass += NewStars[i];
            Gal[p].MetalsStellarMass += NewStarsMetals[i];
            
			if(Gal[p].DiscStarsMetals[i] > Gal[p].DiscStars[i]) printf("DiscStars, Metals = %e, %e\n", Gal[p].DiscStars[i], Gal[p].DiscStarsMetals[i]);
			assert(Gal[p].DiscStarsMetals[i] <= Gal[p].DiscStars[i]);
            
            if(AgeStructOut>0)
            {
                if(Gal[p].DiscStarsAge[i][k_now] + NewStars[i] > 0)
                {
                    Gal[p].VelDispStarsAge[i][k_now] = sqrt( (Gal[p].DiscStarsAge[i][k_now]*sqr(Gal[p].VelDispStarsAge[i][k_now]) + NewStars[i]*sqr(sigma_gas)) / (Gal[p].DiscStarsAge[i][k_now] + NewStars[i])  );
                    assert(Gal[p].VelDispStarsAge[i][k_now] >= 0);
                    Gal[p].DiscStarsAge[i][k_now] += NewStars[i];
                    Gal[p].DiscStarsMetalsAge[i][k_now] += NewStarsMetals[i];
                }
                assert(Gal[p].DiscStarsMetalsAge[i][k_now] <= Gal[p].DiscStarsAge[i][k_now]);
            }
		}
	}
	
    DiscStarSum = get_disc_stars(p);
    
	// Readjust disc to deal with any retrograde stars
	if(cos_angle_sdisc_comb<0.0)
		J_retro = J_sdisc*fabs(cos_angle_sdisc_comb);
	else if(cos_angle_new_comb<0.0)
		J_retro = J_new*fabs(cos_angle_new_comb);
	else
		J_retro = 0.0;
	J_sum = J_sdisc*fabs(cos_angle_sdisc_comb) + J_new*fabs(cos_angle_new_comb);
		
	if(J_retro>0.0)
	{
        
        project_disc_with_dispersion(Gal[p].DiscStars, Gal[p].DiscStarsMetals, Gal[p].VelDispStars, Gal[p].DiscStarsAge, Gal[p].DiscStarsMetalsAge, Gal[p].VelDispStarsAge, (J_sum - 2.0*J_retro)/J_sum, p, k_now, Disc1, Disc1Metals, Disc1VelDisp, Disc1Age, Disc1MetalsAge, Disc1VelDispAge);
        
		for(i=0; i<N_BINS; i++)
		{
			Gal[p].DiscStars[i] = Disc1[i];
			Gal[p].DiscStarsMetals[i] = Disc1Metals[i];
            Gal[p].VelDispStars[i] = Disc1VelDisp[i];
            assert(Gal[p].VelDispStars[i] >= 0);
			assert(Gal[p].DiscStarsMetals[i] <= Gal[p].DiscStars[i]);
            
            if(AgeStructOut>0)
            {
                for(k=k_now; k<N_AGE_BINS; k++)
                {
                    Gal[p].DiscStarsAge[i][k] = Disc1Age[i][k];
                    Gal[p].DiscStarsMetalsAge[i][k] = Disc1MetalsAge[i][k];
                    Gal[p].VelDispStarsAge[i][k] = Disc1VelDispAge[i][k];
                    assert(Gal[p].VelDispStarsAge[i][k] >= 0);
                    assert(Gal[p].DiscStarsMetalsAge[i][k] <= Gal[p].DiscStarsAge[i][k]);
                }
            }
		}
    }
	
	// Set the new spin direction of the stellar disc
	for(i=0; i<3; i++)
    {
		Gal[p].SpinStars[i] = SDiscNewSpin[i];
		assert(Gal[p].SpinStars[i]==Gal[p].SpinStars[i]);
        
    }
    
    DiscStarSum = get_disc_stars(p);
    
    if(DiscStarSum>0.0) assert(DiscStarSum+Gal[p].SecularBulgeMass+Gal[p].ClassicalBulgeMass <= 1.01*Gal[p].StellarMass && DiscStarSum+Gal[p].SecularBulgeMass+Gal[p].ClassicalBulgeMass >= Gal[p].StellarMass/1.01);

	for(i=0; i<N_BINS; i++)
    {
		if(Gal[p].DiscStarsMetals[i] > Gal[p].DiscStars[i]) printf("DiscStars, Metals = %e, %e\n", Gal[p].DiscStars[i], Gal[p].DiscStarsMetals[i]);
		assert(Gal[p].DiscStarsMetals[i] <= Gal[p].DiscStars[i]);
    }

    update_stellardisc_scaleradius(p);
}


void project_disc(double DiscMass[N_BINS], double cos_angle, int p, double *NewDisc)
{
	double high_bound, ratio_last_bin;
	int i, j, j_old, l;
    
	cos_angle = fabs(cos_angle); // This function will not deal with retrograde motion so needs an angle less than pi/2
    
    if(cos_angle > 0.99)
    { // angle is irrelevant, just return the original disc to prevent floating-point nonsense
        for(i=0; i<N_BINS; i++) NewDisc[i] = DiscMass[i];
        return;
    }
	
	j_old = 0;

	for(i=0; i<N_BINS; i++)
	{
		high_bound = DiscBinEdge[i+1] / cos_angle;
		j = j_old;
		
		while(DiscBinEdge[j]<high_bound)
		{
			j++;
			if(j==N_BINS) break;
		} 
		j -= 1;
		
		NewDisc[i] = 0.0;
		for(l=j_old; l<j; l++) 
		{
			NewDisc[i] += DiscMass[l];
			DiscMass[l] = 0.0;
		}
		if(i!=N_BINS-1)
		{
			if(j!=N_BINS-1){
				ratio_last_bin = sqr((high_bound - DiscBinEdge[j]) / (DiscBinEdge[j+1]-DiscBinEdge[j]));
				assert(ratio_last_bin<=1.0);}
			else if(high_bound < Gal[p].Rvir*Gal[p].Vvir){
				ratio_last_bin = sqr((high_bound - DiscBinEdge[j]) / (Gal[p].Rvir*Gal[p].Vvir-DiscBinEdge[j]));
				assert(ratio_last_bin<=1.0);}
			else
				ratio_last_bin = 1.0;
			NewDisc[i] += ratio_last_bin * DiscMass[j];
			DiscMass[j] -= ratio_last_bin * DiscMass[j];
		}
		else
		{
			NewDisc[i] = DiscMass[i];
		}
        if(!(NewDisc[i]>=0.0)) 
        {
            printf("i, NewDisc[i], j, ratio_last_bin, cos_angle = %i, %e, %i, %e, %e\n", i, NewDisc[i], j, ratio_last_bin, cos_angle);
            for(l=0; l<=j; l++) printf("l, DiscMass[l] = %i, %e\n", l, DiscMass[l]);
        }
		assert(NewDisc[i]>=0.0);

		j_old = j;
	}
}


void project_disc_with_dispersion(double DiscMass[N_BINS], double DiscMetals[N_BINS], double VelDisp[N_BINS], double DiscMassAge[N_BINS][N_AGE_BINS], double DiscMetalsAge[N_BINS][N_AGE_BINS], double VelDispAge[N_BINS][N_AGE_BINS], double cos_angle, int p, int k_now, double *NewDisc, double *NewMetals, double *NewVelDisp, double (*NewDiscAge)[N_AGE_BINS], double (*NewMetalsAge)[N_AGE_BINS], double (*NewVelDispAge)[N_AGE_BINS])
{
    
    
    double high_bound, ratio_last_bin=1.0;
    int i, j, j_old, l, k;
    
    cos_angle = fabs(cos_angle); // This function will not deal with retrograde motion so needs an angle less than pi/2
    
    if(cos_angle > 0.99)
    { // angle is irrelevant, just return the original disc to prevent floating-point nonsense
        for(i=0; i<N_BINS; i++) 
        {
            NewDisc[i] = DiscMass[i];
            NewMetals[i] = DiscMetals[i];
            NewVelDisp[i] = VelDisp[i];
            assert(NewVelDisp[i] >= 0);
            
            if(AgeStructOut > 0)
            {
                for(k=k_now; k<N_AGE_BINS; k++)
                {
                    if(!(DiscMetalsAge[i][k] <= DiscMassAge[i][k]))
                        printf("i, k, DiscMetalsAge[i][k], DiscMassAge[i][k] = %i, %i, %e, %e\n", i, k, DiscMetalsAge[i][k], DiscMassAge[i][k]);
                    
                    assert(DiscMetalsAge[i][k] <= DiscMassAge[i][k]);
                    NewDiscAge[i][k] = DiscMassAge[i][k];
                    
                    NewMetalsAge[i][k] = DiscMetalsAge[i][k];
                    NewVelDispAge[i][k] = VelDispAge[i][k];
                    assert(NewVelDispAge[i][k] >= 0);
                    
                    assert(NewMetalsAge[i][k] <= NewDiscAge[i][k]);
                }
            }
        }
        return;
    }
    
    j_old = 0;

    for(i=0; i<N_BINS; i++)
    {
        high_bound = DiscBinEdge[i+1] / cos_angle;
        j = j_old;
        
        while(DiscBinEdge[j]<high_bound)
        {
            j++;
            if(j==N_BINS) break;
        } 
        j -= 1;
        
        NewDisc[i] = 0.0;
        NewMetals[i] = 0.0;
        NewVelDisp[i] = 0.0; // Will treat this as variance and take the square root at the end of the loop to avoid unnecessary repetitive squaring and rooting
        if(AgeStructOut > 0)
        {
            for(k=k_now; k<N_AGE_BINS; k++)
            {
                NewDiscAge[i][k] = 0.0;
                NewMetalsAge[i][k] = 0.0;
                NewVelDispAge[i][k] = 0.0;
            }
        }
        
        for(l=j_old; l<j; l++) 
        {
            if(NewDisc[i] + DiscMass[l] > 0.0)
            {
                NewVelDisp[i] = (NewDisc[i]*NewVelDisp[i] + DiscMass[l]*sqr(VelDisp[l])) / (NewDisc[i] + DiscMass[l]);
                assert(NewVelDisp[i] >= 0);
                NewDisc[i] += DiscMass[l];
                NewMetals[i] += DiscMetals[l];
                
                if(AgeStructOut > 0)
                {
                    for(k=k_now; k<N_AGE_BINS; k++)
                    {
                        if(NewDiscAge[i][k] + DiscMassAge[l][k] > 0)
                        {
                            NewVelDispAge[i][k] = (NewDiscAge[i][k]*NewVelDispAge[i][k] + DiscMassAge[l][k]*sqr(VelDispAge[l][k])) / (NewDiscAge[i][k] + DiscMassAge[l][k]);
                            assert(NewVelDispAge[i][k] >= 0);
                            NewDiscAge[i][k] += DiscMassAge[l][k];
                            NewMetalsAge[i][k] += DiscMetalsAge[l][k];
                            assert(NewMetalsAge[i][k] <= NewDiscAge[i][k]);
                        }
                        else
                            NewVelDispAge[i][k] = 0.0;
                        
                        DiscMassAge[l][k] = 0.0;
                        DiscMetalsAge[l][k] = 0.0;
                    }
                }
            }
            else
                NewVelDisp[i] = 0.0;
            
            DiscMass[l] = 0.0;
            DiscMetals[l] = 0.0;
            
            
            
        }
        if(i!=N_BINS-1)
        {
            if(j!=N_BINS-1){
                ratio_last_bin = sqr((high_bound - DiscBinEdge[j]) / (DiscBinEdge[j+1]-DiscBinEdge[j]));
                assert(ratio_last_bin<=1.0);}
            else if(high_bound < Gal[p].Rvir*Gal[p].Vvir){
                ratio_last_bin = sqr((high_bound - DiscBinEdge[j]) / (Gal[p].Rvir*Gal[p].Vvir-DiscBinEdge[j]));
                assert(ratio_last_bin<=1.0);}
            else
                ratio_last_bin = 1.0;
            
            if(NewDisc[i] + ratio_last_bin*DiscMass[j] > 0)
            {
                NewVelDisp[i] = (NewDisc[i]*NewVelDisp[i] + ratio_last_bin*DiscMass[j]*sqr(VelDisp[j])) / (NewDisc[i] + ratio_last_bin*DiscMass[j]);
                assert(NewVelDisp[i] >= 0);
                NewDisc[i] += ratio_last_bin * DiscMass[j];
                NewMetals[i] += ratio_last_bin * DiscMetals[j];
                
                DiscMass[j] -= ratio_last_bin * DiscMass[j];
                DiscMetals[j] -= ratio_last_bin * DiscMetals[j];
                
                if(AgeStructOut > 0)
                {
                    for(k=k_now; k<N_AGE_BINS; k++)
                    {
                        if(NewDiscAge[i][k] + ratio_last_bin*DiscMassAge[j][k] > 0)
                        {
                            NewVelDispAge[i][k] = (NewDiscAge[i][k]*NewVelDispAge[i][k] + ratio_last_bin*DiscMassAge[j][k]*sqr(VelDispAge[j][k])) / (NewDiscAge[i][k] + ratio_last_bin*DiscMassAge[j][k]);
                            assert(NewVelDispAge[i][k] >= 0);
                            NewDiscAge[i][k] += ratio_last_bin * DiscMassAge[j][k];
                            NewMetalsAge[i][k] += ratio_last_bin * DiscMetalsAge[j][k];
                            assert(NewMetalsAge[i][k] <= NewDiscAge[i][k]);

                            DiscMassAge[j][k] -= ratio_last_bin * DiscMassAge[j][k];
                            DiscMetalsAge[j][k] -= ratio_last_bin * DiscMetalsAge[j][k];
                        }
                        
                        NewVelDispAge[i][k] = sqrt(NewVelDispAge[i][k]); // taking square root now
                        assert(NewVelDispAge[i][k] >= 0);
                    }
                }
            }
            
            NewVelDisp[i] = sqrt(NewVelDisp[i]); // taking square root now
            assert(NewVelDisp[i] >= 0);
            
            
        }
        else
        {
            NewVelDisp[i] = VelDisp[i];
            assert(NewVelDisp[i] >= 0);
            NewDisc[i] = DiscMass[i]; // changing = -> += would have no difference
            NewMetals[i] = DiscMetals[i];
            
            if(AgeStructOut > 0)
            {
                for(k=k_now; k<N_AGE_BINS; k++)
                {
                    NewVelDispAge[i][k] = VelDispAge[i][k];
                    assert(NewVelDispAge[i][k] >= 0);
                    NewDiscAge[i][k] = DiscMassAge[i][k];
                    NewMetalsAge[i][k] = DiscMetalsAge[i][k];
                    assert(NewMetalsAge[i][k] <= NewDiscAge[i][k]);
                }
            }
            
        }
        if(!(NewDisc[i]>=0.0)) 
        {
            printf("i, NewDisc[i], j, ratio_last_bin, cos_angle = %i, %e, %i, %e, %e\n", i, NewDisc[i], j, ratio_last_bin, cos_angle);
            for(l=0; l<=j; l++) printf("l, DiscMass[l] = %i, %e\n", l, DiscMass[l]);
        }
        assert(NewDisc[i]>=0.0);

        j_old = j;
    }
}

void update_HI_H2(int p, double time, int k_now)
{
    double area, f_H2, f_H2_HI=1e-10, Pressure, f_sigma;
    int i;
    double angle = acos(Gal[p].SpinStars[0]*Gal[p].SpinGas[0] + Gal[p].SpinStars[1]*Gal[p].SpinGas[1] + Gal[p].SpinStars[2]*Gal[p].SpinGas[2])*180.0/M_PI;
    double sigma_gas, A_term_global, xi_term_global;//, A_term_annulus, xi_term_annulus, xi_on_2A, full_ratio, interrim;
    double s, Zp, chi, c_f, Sigma_comp0, Tau_c, Sigma_cold;
    double X_H=0.76, Z, f_neutral=1.0;
    
    sigma_gas = (1.1e6 + 1.13e6 * ZZ[Gal[p].SnapNum])/UnitVelocity_in_cm_per_s;
    double sigma_gas_sqr = sqr(sigma_gas);
    
    if(Gal[p].Vvir>0.0 && Gal[p].ColdGas>0.0)
    {
        A_term_global = uni_ion_A / sqr(sigma_gas_sqr);
        xi_term_global = uni_ion_xi / sigma_gas_sqr;
        
        for(i=0; i<N_BINS; i++)
        {
            area = M_PI * (sqr(Gal[p].DiscRadii[i+1]) - sqr(Gal[p].DiscRadii[i]));
            Z = get_metallicity(Gal[p].DiscGas[i], Gal[p].DiscGasMetals[i]);
            Sigma_cold = Gal[p].DiscGas[i] / area;
            
            if(Gal[p].DiscGas[i]<=MIN_STARFORMATION) // imposing a reasonable minimum to avoid silly ratios triggering the code to crash 
            {
                Gal[p].DiscGas[i] = 0.0;
                Gal[p].DiscGasMetals[i] = 0.0;
                Gal[p].DiscHI[i] = 0.0;
                Gal[p].DiscH2[i] = 0.0;
                continue;
            }
            assert(Gal[p].DiscGas[i]>0.0);
            
            if(H2prescription==1 || H2prescription==2)
            {
                Zp = Z / 0.0142; // Might also want solar metal fraction to be variable too
                
                if(Zp>0.01 && Zp<1) // Fu et al. 2013
                    c_f = ClumpFactor*pow(Zp, -ClumpExponent);
                else if(Zp>=1)
                    c_f = ClumpFactor;
                else // prescription from Fu not defined here, so assume clumping factor can't exceed this value ~25
                    c_f = ClumpFactor*pow(0.01, -ClumpExponent);
                
                Sigma_comp0 = c_f * Gal[p].DiscGas[i]/area; // see Krumholz & Dekel 2012, originally comes from McKee & Krumholz 2010
                Tau_c = 320 * Zp * Sigma_comp0 * UnitMass_in_g / sqr(UnitLength_in_cm) * Hubble_h;
                chi = 3.1 * (1+ 3.1*pow(Zp,0.365)) / 4.1;
                s = log(1 + 0.6*chi + 0.01*chi*chi) / (0.6*Tau_c);
                if(s<2)
                {
                    f_H2 = 1.0 - 0.75*s/(1+0.25*s); // Not actual H2/cold, but rather H2/(H2+HI)
                    f_H2_HI = 1.0 / (1.0/f_H2 - 1.0);
                }
                else
                    f_H2_HI = 0.0;
                
            }
            
            if(H2prescription==0 || H2prescription==2)
            {
                if(angle <= ThetaThresh)
                {
                    f_sigma =  sigma_gas / Gal[p].VelDispStars[i];
                    Pressure = 0.5*M_PI*G * Gal[p].DiscGas[i] * (Gal[p].DiscGas[i] + f_sigma*Gal[p].DiscStars[i]) / sqr(area) * Hubble_h * Hubble_h;
                }
                else
                    Pressure = 0.5*M_PI*G * sqr(Gal[p].DiscGas[i]/area) * Hubble_h * Hubble_h;
                
                if(H2prescription==2)
                    f_H2_HI = dmax(f_H2_HI, H2FractionFactor * pow(Pressure/P_0, H2FractionExponent));
                else
                    f_H2_HI = H2FractionFactor * pow(Pressure/P_0, H2FractionExponent);

            }

            double SFR_guess = 0.0;
            
            if(Gal[p].DiscStarsAge[i][k_now] > 0.0)
                SFR_guess = Gal[p].DiscStarsAge[i][k_now] / ((1.0 - RecycleFraction) * (AgeBinEdge[k_now+1] - time));
            else if(k_now < N_AGE_BINS-1)
            {
                if(Gal[p].DiscStarsAge[i][k_now+1] > 0.0)
                {
                    // need to recalculate the relevant returned-mass fraction here
                    double StellarOutput[2];
                    get_RecycleFraction_and_NumSNperMass(time, 0.5*(AgeBinEdge[k_now+2]+AgeBinEdge[k_now+1]), StellarOutput);
                    double ReturnFraction = StellarOutput[0];
                    SFR_guess = Gal[p].DiscStarsAge[i][k_now+1] / ((1.0 - ReturnFraction) * (AgeBinEdge[k_now+2] - AgeBinEdge[k_now+1]));
                }
                    
            }
            
            
            // neutral fraction
            // can undo the SFR assumption in earlier formula, then reapply as an idea
            f_neutral = 1.0 - sqrt(uni_ion_hist * (3*X_H+1) * sqr(sigma_gas * area / X_H) * SFR_guess / cube(Gal[p].DiscGas[i]));
            
            // calculate maximum neutral fraction based on the cosmic ionizing background radiation
            double gamma_term = uni_fneut_bg * (3*X_H+1) * GammaHI_z[Gal[p].SnapNum] / X_H * sqr(sigma_gas * area / Gal[p].DiscGas[i]);
            double f_neutral_max = dmax(1.0 + gamma_term - sqrt(sqr(gamma_term) + 2*gamma_term), 0.0);
            if(f_neutral > f_neutral_max)
            {
                f_neutral = f_neutral_max;
            }
                
            if(H2prescription==3)
            {
                // THIS IS NOT THE AVERAGE DISTANCE FROM A DISTRIBUTED SOURCE. AN INTEGRAL IS REQUIRED.  Could approximate as a 1D integral around the centre of the annulus?
                double av_dist = sqrt(area);
                double Sbar = av_dist * 1e4/Hubble_h;
                double Sbar5 = Sbar*Sbar*Sbar*Sbar*Sbar;
                double Dstar = 0.17 * (2.0 + Sbar5) / (1.0 + Sbar5);
                double DMW2 = sqr(Z/0.0127);
                double gbar = sqrt(sqr(Dstar) + DMW2);
                double UMW, alphabar, Sigma_R1;

                UMW = dmax(UVB_z[Gal[p].SnapNum], UVMW_perSFRdensity * SFR_guess / area); 
                alphabar = 0.5 + 1.0 / (1.0 + sqrt(UMW * DMW2 / 600.0));
                Sigma_R1 = Sigma_R1_fac * sqrt(0.001 + 0.1*UMW) / (gbar * (1.0 + 1.69*sqrt(0.001 + 0.1*UMW)));
                f_H2_HI = pow(f_neutral * X_H * Gal[p].DiscGas[i] / (Sigma_R1 * area), alphabar);
            }
                // reset HI and H2 values based on GD14 prescription
                if(f_H2_HI > 0.0)
                {
                    f_H2 = X_H / (1.0/f_H2_HI + 1);
                    Gal[p].DiscH2[i] = f_H2 * f_neutral * Gal[p].DiscGas[i];
                    Gal[p].DiscHI[i] = X_H * f_neutral * Gal[p].DiscGas[i] - Gal[p].DiscH2[i];
                    if(Gal[p].DiscHI[i] < 0.0) Gal[p].DiscHI[i] = 0.0;
                }
                else
                {
                    Gal[p].DiscH2[i] = 0.0;
                    Gal[p].DiscHI[i] = X_H * f_neutral * Gal[p].DiscGas[i];
                }            
        }
    }
    else if (Gal[p].ColdGas <= 0.0)
    {
        for(i=0; i<N_BINS; i++)
        {
            Gal[p].DiscGas[i] = 0.0;
            Gal[p].DiscGasMetals[i] = 0.0;
            Gal[p].DiscHI[i] = 0.0;
            Gal[p].DiscH2[i] = 0.0;
        }
    }
}



void delayed_feedback(int p, int k_now, double time, double dt)
{
    int k, i;
    double t0, t1, metallicity=BIG_BANG_METALLICITY, return_mass, energy_feedback, annulus_radius, annulus_velocity, cold_specific_energy, reheat_specific_energy, reheated_mass, hot_specific_energy, return_metal_mass, new_metals, j_hot, hot_thermal_and_kinetic, eject_specific_energy, escape_velocity2, ejected_cold_mass, norm_ratio, reheat_eject_sum;
    double v_therm2, vertical_velocity, energy_onto_hot, returned_mass_hot, returned_mass_cold, ejected_sum;
    double StellarOutput[2];
    double v_wind, new_ejected_specific_energy=0.0;

    double inv_FinalRecycleFraction = 1.0/FinalRecycleFraction;
    double tdyn = 0.1 / sqrt(Hubble_sqr_z(Halo[Gal[p].HaloNr].SnapNum));
    
    assert(Gal[p].MetalsHotGas<=Gal[p].HotGas);
    assert(Gal[p].OutflowGas >= 0.0);
    assert(Gal[p].MetalsHotGas>=0);
  
    j_hot = 2 * Gal[p].Vvir * Gal[p].CoolScaleRadius;
    hot_thermal_and_kinetic = 0.5 * (sqr(Gal[p].Vvir) + sqr(j_hot)/Gal[p].R2_hot_av);
    hot_specific_energy = Gal[p].HotGasPotential + hot_thermal_and_kinetic;

    
    
    // loop over age bins prior to the current one and calculate the return fraction and SN per remaining mass
    // might be more optimal to do this calculation outside this function and feed it in.  Or, even better, build a look-up table at the start of running Dark Sage.  The same calculation is being done many times with the current set-up, which is redundant.  The logic right now is safe though!
    if(k_now==N_AGE_BINS-1) return;
    double ReturnFraction[N_AGE_BINS], SNperMassRemaining[N_AGE_BINS];
//    double time_convert = 1e-3 * UnitTime_in_s / SEC_PER_MEGAYEAR / Hubble_h;
    for(k=N_AGE_BINS-1; k>k_now; k--)
    {
        t0 = 0.5*(AgeBinEdge[k]+AgeBinEdge[k+1]) - (time+0.5*dt); // start of time window considered for the delayed feedback
        if(!(t0>0)) printf("k, AgeBinEdge[k-1], AgeBinEdge[k], AgeBinEdge[k+1], time+0.5*dt  = %i, %e, %e, %e, %e", k, AgeBinEdge[k-1], AgeBinEdge[k], AgeBinEdge[k+1], time+0.5*dt);
        assert(t0>0);
        t1 = t0 + dt; // end of time window
        assert(t1>t0);
        get_RecycleFraction_and_NumSNperMass(t0, t1, StellarOutput);
        ReturnFraction[k] = StellarOutput[0];
        SNperMassRemaining[k] = StellarOutput[1];
        assert(ReturnFraction[k]>=0);
        assert(ReturnFraction[k]<FinalRecycleFraction);
    }
        
    energy_onto_hot = 0.0; // add to this field to find the total energy directly imparted onto the CGM from spheroid stars
    returned_mass_hot = 0.0; // total mass returned directly into the CGM from spheroid stars
            
    // capture the energy from feedback associated with bulge stars and add stellar mass loss to hot component
    for(k=N_AGE_BINS-1; k>k_now; k--)
    {
        // deal with stellar outflow from merger-driven bulge
        if(Gal[p].ClassicalBulgeMassAge[k]>0)
        {
            energy_onto_hot += (Gal[p].ClassicalBulgeMassAge[k] * EnergySNcode * SNperMassRemaining[k]); // energy from feedback of these stars goes to ejecting hot gas
            metallicity = get_metallicity(Gal[p].ClassicalBulgeMassAge[k], Gal[p].ClassicalMetalsBulgeMassAge[k]);
            
            return_mass = ReturnFraction[k] * Gal[p].ClassicalBulgeMassAge[k];
            return_metal_mass = ReturnFraction[k] * Gal[p].ClassicalMetalsBulgeMassAge[k];
            returned_mass_hot += return_mass;
            
            Gal[p].ClassicalBulgeMassAge[k] -= return_mass;
            Gal[p].ClassicalBulgeMass -= return_mass;
            Gal[p].StellarMass -= return_mass;
            assert(Gal[p].ClassicalBulgeMass >= 0);

            Gal[p].ClassicalMetalsBulgeMassAge[k] -= return_metal_mass;
            Gal[p].ClassicalMetalsBulgeMass -= return_metal_mass;
            Gal[p].MetalsStellarMass -= return_metal_mass;
            
            if(Gal[p].FountainGas + return_mass > 0.0)
                Gal[p].FountainTime = (Gal[p].FountainGas * Gal[p].FountainTime + return_mass * tdyn) / (Gal[p].FountainGas + return_mass);
            Gal[p].FountainGas += return_mass;
            Gal[p].MetalsFountainGas += return_metal_mass;
            Gal[p].MetalsFountainGas += (inv_FinalRecycleFraction * return_mass * Yield * (1-metallicity)); // enrich gas with new metals from this stellar population
            
            // ex-situ stars are part of the merger-driven bulge.  Adjust their mass too.
            return_mass = ReturnFraction[k] * Gal[p].StarsExSituAge[k];
            return_metal_mass = ReturnFraction[k] * Gal[p].MetalsStarsExSituAge[k];
            Gal[p].StarsExSituAge[k] -= return_mass;
            Gal[p].StarsExSitu -= return_mass;
            Gal[p].MetalsStarsExSituAge[k] -= return_metal_mass;
            Gal[p].MetalsStarsExSitu -= return_metal_mass;
        }
        
        // same for instability-driven bulge
        if(Gal[p].SecularBulgeMassAge[k]>0)
        {
            energy_onto_hot += (Gal[p].SecularBulgeMassAge[k] * EnergySNcode * SNperMassRemaining[k]);
            metallicity = get_metallicity(Gal[p].SecularBulgeMassAge[k], Gal[p].SecularMetalsBulgeMassAge[k]);
            
            return_mass = ReturnFraction[k] * Gal[p].SecularBulgeMassAge[k];
            return_metal_mass = ReturnFraction[k] * Gal[p].SecularMetalsBulgeMassAge[k];
            returned_mass_hot += return_mass;
            
            Gal[p].SecularBulgeMassAge[k] -= return_mass;
            Gal[p].SecularBulgeMass -= return_mass;
            Gal[p].StellarMass -= return_mass;
            assert(Gal[p].SecularBulgeMass >= 0);
            
            Gal[p].SecularMetalsBulgeMassAge[k] -= return_metal_mass;
            Gal[p].SecularMetalsBulgeMass -= return_metal_mass;
            Gal[p].MetalsStellarMass -= return_metal_mass;
            
            if(Gal[p].FountainGas + return_mass > 0.0)
                Gal[p].FountainTime = (Gal[p].FountainGas * Gal[p].FountainTime + return_mass * tdyn) / (Gal[p].FountainGas + return_mass);
            Gal[p].FountainGas += return_mass;
            Gal[p].MetalsFountainGas += return_metal_mass;
            Gal[p].MetalsFountainGas += (inv_FinalRecycleFraction * return_mass * Yield * (1-metallicity));
        }

        // same for intracluster stars
        if(Gal[p].ICS_Age[k]>0)
        {
            energy_onto_hot += (Gal[p].ICS_Age[k] * EnergySNcode * SNperMassRemaining[k]);
            metallicity = get_metallicity(Gal[p].ICS_Age[k], Gal[p].MetalsICS_Age[k]);
            
            return_mass = ReturnFraction[k] * Gal[p].ICS_Age[k];
            return_metal_mass = ReturnFraction[k] * Gal[p].MetalsICS_Age[k];
            returned_mass_hot += return_mass;
            
            Gal[p].ICS_Age[k] -= return_mass;
            Gal[p].ICS -= return_mass;
            assert(Gal[p].ICS >= 0);

            Gal[p].MetalsICS_Age[k] -= return_metal_mass;
            Gal[p].MetalsICS -= return_metal_mass;
            
            if(Gal[p].FountainGas + return_mass > 0.0)
                Gal[p].FountainTime = (Gal[p].FountainGas * Gal[p].FountainTime + return_mass * tdyn) / (Gal[p].FountainGas + return_mass);
            Gal[p].FountainGas += return_mass;
            Gal[p].MetalsFountainGas += return_metal_mass;
            Gal[p].MetalsFountainGas += (inv_FinalRecycleFraction * return_mass * Yield * (1-metallicity));
        }
        
        // and for local intergalactic stars
        if(Gal[p].LocalIGS_Age[k] > 0)
        {
            return_mass = ReturnFraction[k] * Gal[p].LocalIGS_Age[k];
            return_metal_mass = ReturnFraction[k] * Gal[p].MetalsLocalIGS_Age[k];
            
            Gal[p].LocalIGS_Age[k] -= return_mass;
            Gal[p].LocalIGS -= return_mass;
            
            Gal[p].MetalsLocalIGS_Age[k] -= return_metal_mass;
            Gal[p].MetalsLocalIGS -= return_metal_mass;
            
            Gal[p].LocalIGM += return_mass;
            Gal[p].MetalsLocalIGM += return_metal_mass;
            Gal[p].MetalsLocalIGM += (inv_FinalRecycleFraction * return_mass * Yield * (1-metallicity));
        }
        
    }
    
    
    if((energy_onto_hot>0) && (Gal[p].OutflowSpecificEnergy <= hot_specific_energy))
    {
        Gal[p].OutflowSpecificEnergy = Gal[p].EjectedPotential + hot_thermal_and_kinetic; // set as the minimum
        assert(Gal[p].OutflowSpecificEnergy == Gal[p].OutflowSpecificEnergy);
    }

    
    // loop over disc annuli
    for(i=0; i<N_BINS; i++)
    {
        // initialise for this annulus
        energy_feedback = 0.0;
        returned_mass_cold = 0.0;

        
        // loop over age bins for an annulus
        for(k=N_AGE_BINS-1; k>k_now; k--)
        {
            // ensure there is something to actually do
            if(SNperMassRemaining[k]<=0 || ReturnFraction[k]<=0 || Gal[p].DiscStarsAge[i][k]<=0) continue;
            
            // initial calculation of reheated mass from supernovae
            energy_feedback += (Gal[p].DiscStarsAge[i][k] * EnergySNcode * SNperMassRemaining[k]);

            // stellar mass loss
            metallicity = get_metallicity(Gal[p].DiscStarsAge[i][k], Gal[p].DiscStarsMetalsAge[i][k]);
            return_mass = ReturnFraction[k] * Gal[p].DiscStarsAge[i][k];
            return_metal_mass = ReturnFraction[k] * Gal[p].DiscStarsMetalsAge[i][k];
            returned_mass_cold += return_mass;
            
            Gal[p].DiscStars[i] -= return_mass;
            Gal[p].DiscStarsAge[i][k] -= return_mass;
            Gal[p].StellarMass -= return_mass;
            
            Gal[p].DiscStarsMetals[i] -= return_metal_mass;
            Gal[p].DiscStarsMetalsAge[i][k] -= return_metal_mass;
            Gal[p].MetalsStellarMass -= return_metal_mass;
            
            Gal[p].DiscGas[i] += return_mass;
            Gal[p].DiscGasMetals[i] += return_metal_mass;
            Gal[p].ColdGas += return_mass;
            Gal[p].MetalsColdGas += return_metal_mass;

            assert(Gal[p].DiscStarsAge[i][k]>0);
            if(!(Gal[p].DiscStarsMetalsAge[i][k]>=0)) printf("Gal[p].DiscStarsAge[i][k], Gal[p].DiscStarsMetalsAge[i][k], metallicity, return_metal_mass = %e, %e, %e, %e\n", Gal[p].DiscStarsAge[i][k], Gal[p].DiscStarsMetalsAge[i][k], metallicity, return_metal_mass);
            assert(Gal[p].DiscStarsMetalsAge[i][k]>=0);
            
            // enrich gas with new metals from this stellar population
            new_metals = (inv_FinalRecycleFraction * return_mass * Yield * (1-metallicity));
            Gal[p].DiscGasMetals[i] += new_metals;
            Gal[p].MetalsColdGas += new_metals;
        }
        
        if(energy_feedback<=0 || returned_mass_cold <= MIN_STARFORMATION) continue;
        
        
        // relevant energy quantities for this annulus
        annulus_radius = sqrt(0.5 * (sqr(Gal[p].DiscRadii[i]) + sqr(Gal[p].DiscRadii[i+1])) );
        annulus_velocity = 0.5 * (DiscBinEdge[i] + DiscBinEdge[i+1]) / annulus_radius;
        vertical_velocity = (1.1e6 + 1.13e6 * ZZ[Gal[p].SnapNum])/UnitVelocity_in_cm_per_s;
        cold_specific_energy = 0.5*(sqr(annulus_velocity) + sqr(vertical_velocity) + Gal[p].Potential[i] + Gal[p].Potential[i+1]);
        reheat_specific_energy = hot_specific_energy - cold_specific_energy;

        v_therm2 = sqr(Gal[p].Vvir);
        double v_wind2 = 2.0*reheat_specific_energy - v_therm2;
        v_wind = sqrt(v_wind2);
        escape_velocity2 = 2.0 * (Gal[p].EjectedPotential - Gal[p].Potential[i]);
        
        if(reheat_specific_energy < 0.0) // possible to happen.  Means ejecting is the only logical choice but can't trust v_wind now
        {
            new_ejected_specific_energy = dmax(Gal[p].OutflowSpecificEnergy, dmax(Gal[p].EjectedPotential, cold_specific_energy) + 0.5*v_therm2);
            ejected_cold_mass = energy_feedback / (new_ejected_specific_energy - cold_specific_energy);
            reheated_mass = 0.0;
        }
        else if(sqr(v_wind - annulus_velocity) > escape_velocity2) // the wind speed is high enough to eject all the gas instead of just reheating it
        {
            reheated_mass = 0.0;
            new_ejected_specific_energy = dmax(cold_specific_energy + 0.5*(v_therm2 + v_wind2), Gal[p].EjectedPotential + 0.5*v_therm2);
            ejected_cold_mass = energy_feedback / (new_ejected_specific_energy - cold_specific_energy);
        }
        else if(sqr(v_wind + annulus_velocity) > escape_velocity2) // the wind speed is enough to eject some of the gas
        {
            double reheated_ejected_ratio = 1.0 / (0.5 - (escape_velocity2 - sqr(annulus_velocity) - sqr(v_wind)) / (4.0 * annulus_velocity * v_wind) ) - 1.0;
            new_ejected_specific_energy = 0.25*(Gal[p].Potential[i] + Gal[p].Potential[i+1] + sqr(v_wind + annulus_velocity)) + 0.5*(Gal[p].EjectedPotential + v_therm2);
            ejected_cold_mass = energy_feedback / (reheated_ejected_ratio * reheat_specific_energy + new_ejected_specific_energy - cold_specific_energy);
            reheated_mass = reheated_ejected_ratio * ejected_cold_mass;
        }
        else
        {
            reheated_mass = energy_feedback / reheat_specific_energy;
            ejected_cold_mass = 0.0;
        }
        
        reheat_eject_sum = reheated_mass + ejected_cold_mass;
        if(!(reheat_eject_sum>=0)) 
        { // hit a rare case where this gets triggered and not sure why, but seems entirely inconsequential
            printf("reheated_mass, ejected_cold_mass = %e, %e\n", reheated_mass, ejected_cold_mass);
            reheated_mass = ejected_cold_mass = reheat_eject_sum = 0.0;
        }
        assert(reheat_eject_sum>=0);
        
        if(reheat_eject_sum >= Gal[p].DiscGas[i])
        {
            norm_ratio = Gal[p].DiscGas[i] / reheat_eject_sum;
            reheated_mass *= norm_ratio;
            ejected_cold_mass *= norm_ratio;
            reheat_eject_sum *= norm_ratio;
        }
        else
            norm_ratio = 1.0;
        
        assert(reheated_mass>=0);
        
        // update the specific energy of the outflowing reservoir (where the stuff to be ejected goes first)
        if(ejected_cold_mass > 0.0)
        {
            if(!(new_ejected_specific_energy - Gal[p].Potential[0] - 0.5*sqr(Gal[p].Vvir) > 0.0))
            {
                printf("new_ejected_specific_energy, Gal[p].Potential[0], 0.5*sqr(Gal[p].Vvir) = %e, %e, %e\n", new_ejected_specific_energy, Gal[p].Potential[0], 0.5*sqr(Gal[p].Vvir));
                printf("reheated_mass, ejected_cold_mass = %e, %e\n", reheated_mass, ejected_cold_mass);
                printf("v_wind = %e\n", v_wind);
                printf("sqr(annulus_velocity + v_wind), escape_velocity2, sqr(annulus_velocity - v_wind) = %e, %e, %e\n", sqr(annulus_velocity + v_wind), escape_velocity2, sqr(annulus_velocity - v_wind));
            }

            
            assert(new_ejected_specific_energy - Gal[p].Potential[0] - 0.5*sqr(Gal[p].Vvir) > 0.0);
            assert(Gal[p].OutflowGas + ejected_cold_mass > 0.0);
            Gal[p].OutflowSpecificEnergy = (Gal[p].OutflowGas * Gal[p].OutflowSpecificEnergy + ejected_cold_mass * new_ejected_specific_energy) / (Gal[p].OutflowGas + ejected_cold_mass);
            assert(Gal[p].OutflowSpecificEnergy == Gal[p].OutflowSpecificEnergy);
            update_outflow_time(p, ejected_cold_mass, new_ejected_specific_energy);
        }
 
        // apply feedback
        metallicity = get_metallicity(Gal[p].DiscGas[i], Gal[p].DiscGasMetals[i]);
        
        assert(Gal[p].MetalsHotGas>=0);
        assert(Gal[p].MetalsHotGas<=Gal[p].HotGas);
        assert(Gal[p].OutflowGas >= 0.0);
        update_from_feedback(p, reheated_mass, metallicity, i, ejected_cold_mass);
        
        // there could be excess energy from delayed feedback in principle; gets added to energy dumped onto hot stuff
        energy_onto_hot += ((1.0 - norm_ratio) * energy_feedback);
    }
    
    // below, ejected_sum is the amount of gas ejected from the CGM from spheroid-star feedback
    // assume the gas ejected from this will end up with the same specific energy as the stuff that was ejected from the ISM.
    // if there's nothing to eject, there's nothing to be done with the energy UNLESS I USE IT TO INCREASE THE SPECIFIC ENERGY OF THE EJECTED RESERVOIR
    if(Gal[p].HotGas + Gal[p].FountainGas > 0.0)
    { 
        if(Gal[p].OutflowGas > 0.0)
        {
            eject_specific_energy = dmax(Gal[p].OutflowSpecificEnergy, Gal[p].EjectedPotential + hot_thermal_and_kinetic);
            assert(eject_specific_energy - hot_specific_energy > 0.0);
            ejected_sum = energy_onto_hot / (eject_specific_energy - hot_specific_energy);
            assert(eject_specific_energy != INFINITY);

            if(ejected_sum > Gal[p].HotGas + Gal[p].FountainGas)
            {
                ejected_sum = Gal[p].HotGas + Gal[p].FountainGas;
                assert(ejected_sum > 0.0);
                assert(energy_onto_hot != INFINITY);
                assert(hot_specific_energy != INFINITY);
                eject_specific_energy = energy_onto_hot / ejected_sum + hot_specific_energy;
            }
            
            assert(Gal[p].OutflowGas + ejected_sum > 0.0);
            assert(Gal[p].OutflowSpecificEnergy == Gal[p].OutflowSpecificEnergy);
            
            
            
            assert(eject_specific_energy != INFINITY);

            Gal[p].OutflowSpecificEnergy = (Gal[p].OutflowGas * Gal[p].OutflowSpecificEnergy + ejected_sum * eject_specific_energy) / (Gal[p].OutflowGas + ejected_sum);
            
            if(!(Gal[p].OutflowSpecificEnergy == Gal[p].OutflowSpecificEnergy))
            {
                printf("Gal[p].OutflowSpecificEnergy = %e\n", Gal[p].OutflowSpecificEnergy);
                printf("Gal[p].OutflowGas, ejected_sum = %e, %e\n", Gal[p].OutflowGas, ejected_sum);
                printf("eject_specific_energy = %e\n", eject_specific_energy);
                printf("Gal[p].EjectedPotential, hot_thermal_and_kinetic = %e, %e\n", Gal[p].EjectedPotential, hot_thermal_and_kinetic);
                printf("energy_onto_hot = %e\n", energy_onto_hot);
            }
            assert(eject_specific_energy != INFINITY);
            assert(Gal[p].OutflowSpecificEnergy == Gal[p].OutflowSpecificEnergy);
        }
        else
        { 
            eject_specific_energy = Gal[p].EjectedPotential + hot_thermal_and_kinetic;
            if(!(eject_specific_energy > hot_specific_energy))
            {
                printf("eject_specific_energy, hot_specific_energy = %e, %e\n", eject_specific_energy, hot_specific_energy);
                printf("hot_thermal_and_kinetic, Gal[p].EjectedPotential, Gal[p].HotGasPotential = %e, %e, %e\n", hot_thermal_and_kinetic, Gal[p].EjectedPotential, Gal[p].HotGasPotential);
                printf("Gal[p].Vvir, j_hot, Gal[p].R2_hot_av = %e, %e, %e\n", Gal[p].Vvir, j_hot, Gal[p].R2_hot_av);
                printf("Gal[p].CoolScaleRadius = %e\n", Gal[p].CoolScaleRadius);
            }
            assert(eject_specific_energy > hot_specific_energy);
            ejected_sum = energy_onto_hot / (eject_specific_energy - hot_specific_energy);
            
            if(ejected_sum > Gal[p].HotGas + Gal[p].FountainGas)
            {
                ejected_sum = Gal[p].HotGas + Gal[p].FountainGas;
                assert(ejected_sum > 0.0);
                eject_specific_energy = energy_onto_hot / ejected_sum + hot_specific_energy;
            }
            
            Gal[p].OutflowSpecificEnergy = eject_specific_energy;
            assert(Gal[p].OutflowSpecificEnergy == Gal[p].OutflowSpecificEnergy);
        }
        
        assert(Gal[p].OutflowSpecificEnergy == Gal[p].OutflowSpecificEnergy);
        update_from_ejection(p, ejected_sum);
    }
    
    update_stellar_dispersion(p);
    
    
}


void update_outflow_time(int p, double new_mass, double new_specific_energy)
{
    if(new_mass <= 0.0) return;
    
    double outflow_kinetic_initial, outflow_kinetic_final, new_time;
    
    // assume kinetic energy is initially radial and the gas must move from R=0 to R=Rvir to have outflowed
    outflow_kinetic_initial = new_specific_energy - Gal[p].Potential[0] - 0.5*sqr(Gal[p].Vvir);
    outflow_kinetic_final = new_specific_energy - Gal[p].EjectedPotential - 0.5*sqr(Gal[p].Vvir);
    if(outflow_kinetic_final < 0.0) outflow_kinetic_final = 0.0;
    
    // assume an average radial speed is the average of the above initial and final values
    new_time = 1.414 * Gal[p].Rvir / (sqrt(outflow_kinetic_initial) + sqrt(outflow_kinetic_final));
    Gal[p].OutflowTime = (Gal[p].OutflowGas * Gal[p].OutflowTime + new_mass * new_time) / (Gal[p].OutflowGas + new_mass);

    if(!(Gal[p].OutflowTime >= 0.0))
    {
        printf("new_mass, new_specific_energy = %e, %e\n", new_mass, new_specific_energy);
        printf("Gal[p].OutflowGas, Gal[p].OutflowTime = %e, %e\n", Gal[p].OutflowGas, Gal[p].OutflowTime);
        printf("new_time = %e\n", new_time);
        printf("outflow_kinetic_initial, outflow_kinetic_final = %e, %e\n", outflow_kinetic_initial, outflow_kinetic_final);
        printf("Gal[p].Potential[0], Gal[p].EjectedPotential,  0.5*sqr(Gal[p].Vvir) = %e, %e, %e\n", Gal[p].Potential[0], Gal[p].EjectedPotential,  0.5*sqr(Gal[p].Vvir));
    }
    assert(Gal[p].OutflowTime >= 0.0);
}
