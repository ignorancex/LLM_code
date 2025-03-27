import unittest
import numpy as np
import pandas as pd
import src as odyn

class TestSimulation(unittest.TestCase):

    def test_model(self):

        geo_df = odyn.get_county_mapping_data(county = "Multnomah", state = "OR")
        hesitancy_dict = odyn.get_hesitancy_dict(geo_df)
        prob = list(hesitancy_dict.values())
        self.assertEqual(int(np.sum(prob)), 1)

        # Load Model.
        model = odyn.OpinionNetworkModel(
                            probabilities = prob,
                            importance_of_weight = 5,
                            importance_of_distance = 2.5,
                           )
        self.assertTrue(model.include_opinion)
        self.assertTrue(model.include_weight)
        
        #Populate model.
        model.populate_model(num_agents = 5)
        self.assertEqual(model.belief_df.shape[0],5)

        # Run simulation.
        sim = odyn.NetworkSimulation()
        sim.run_simulation(model = model, stopping_thresh = 2)
        
        stopping_df = pd.DataFrame()
        for i in range(sim.dynamic_belief_df.shape[1] -1):
            stopping_df[i] = sim.dynamic_belief_df.iloc[:,i+1] - sim.dynamic_belief_df.iloc[:,i]
        df = stopping_df.rolling(window = 5, axis = 1).mean().iloc[:,-1]
        self.assertTrue(df.abs().max() < 1)

if __name__ == '__main__':
    unittest.main()