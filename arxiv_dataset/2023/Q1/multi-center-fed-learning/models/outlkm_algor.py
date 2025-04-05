import numpy as np
import copy
from scipy.spatial import distance
from sklearn.cluster import KMeans

class OutlierKmeansAlgor():
    
    def __init__(self, num_points, dimensions, num_clusters, max_iter, seed, init_type='random',
                threshold_dis = 10., threshold_criterion = 0.01, 
                 max_no_improvement = 3, num_part_of_closest_points = 0.9,
                percent_tosample = 0.3):
        self.num_points = num_points
        self.dimensions = dimensions
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.random_state = seed
        self.init_type = init_type
        self.threshold_dis = threshold_dis
        self.threshold_criterion = threshold_criterion
        self.max_no_improvement = max_no_improvement
        self.num_part_of_closest_points = num_part_of_closest_points
        self.percent_tosample = percent_tosample
        self._lables = []
        
        self.labels_ = None
        self.previous_centers = None
        self.k_means = None
        self.no_improvement = 0
        self.finalized = False
        
    def fit(self, points):
        all_points = copy.copy(points)
        num_sample = np.int(np.floor((len(points) * self.percent_tosample)))
        idx = np.random.randint(len(points), size = num_sample)
        points = all_points[idx]
        if self.k_means is None:
            self.init_bige(points)
            self.k_means = KMeans(init=self.init_type, n_clusters = self.num_clusters,
                                 n_init=50,
                                 max_iter = self.max_iter, random_state = self.random_state)
            
        
        self.k_means.fit(points - self.big_E)
        self.sovl_ol_problem(points)
        centers = self.k_means.cluster_centers_
        if self.previous_centers is not None:
            delta =  centers - self.previous_centers
            #print("delta :", delta)
            if np.sqrt(np.sum( (np.array(centers) - np.array(self.previous_centers)) ** 2 ))  < self.threshold_criterion:
                #print("cluster center is not improving")
                self.no_improvement += 1
            else:
                self.no_improvement = 0
        self.previous_centers = centers
        self._lables = self.k_means.predict(all_points)
        
        #check if we stop earlier, invoker will have to decide fit or not
        if self.no_improvement >= self.max_no_improvement:
            self.finalized = True
        
        
    def init_bige(self, points):
        num_sample = len(points)
        self.big_E = np.zeros((num_sample, self.dimensions))        
        mu = np.mean(points, axis=0)
        point_dis = np.apply_along_axis(lambda i: distance.euclidean(i, mu), 1, points)
        copy_point_dis = copy.copy(point_dis)
        copy_point_dis.sort()
        # retrieve 90% closest elements 
        idx = np.int(np.floor(len(points) * self.num_part_of_closest_points))
        init_out_of_clus_distance = copy_point_dis[idx]
        for i in range(len(points)):
            if point_dis[i] > init_out_of_clus_distance:
                self.big_E[i] = points[i]
            
        return
    
    def sovl_ol_problem(self, points):
        centers = self.k_means.cluster_centers_
        kmeans_labels = self.k_means.labels_
        for i in range(len(points)):
            x_center = centers[kmeans_labels[i]]
            temp_ei = np.array(points[i] - x_center)
            term = max(0,  1- self.threshold_dis / max(0.01, distance.euclidean(points[i], x_center)) )
            self.big_E[i] = temp_ei * term 
        return
    
    @property
    def labels(self):
        return self._lables
    
    @property
    def get_all_members(self):
        return {i: np.where(self._lables == i)[0] for i in range(self.num_clusters)}        
    
    
        
 