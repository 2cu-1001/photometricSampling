import open3d
import numpy as np
from sklearn.preprocessing import StandardScaler
from pointAligner import PointAligner


class PcdEvaluator:
    def __init__(self, pcds):
        self.pcds = pcds
        self.scores = np.array([])
        self.pcds_cnt = len(pcds)
        self.tot_scores = np.array([0] * self.pcds_cnt)

        for i in range(self.pcds_cnt):
            cur_dict = {"points_count" : -1, "noise" : -1, "density" : -1}
            self.scores = np.append(self.scores, cur_dict)
    

    def eval_pcd_score(self):
        for i in range(self.pcds_cnt):
            cur_pcd = self.pcds[i]
            cur_points = self.pcds[i]
            pointAlinger = PointAligner()
            cur_pcd = pointAlinger.numpy_to_pcd(cur_pcd)
            self.scores[i]["points_count"] = len(cur_points)
            
            _, ind = cur_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)
            noise = (len(cur_pcd.points) - len(ind)) / len(cur_pcd.points)
            self.scores[i]["noise"] = noise

            dist = cur_pcd.compute_nearest_neighbor_distance()
            density = 1 / np.mean(dist)
            self.scores[i]["density"] = density  
            print(f"evaluate pcd : {i + 1}/{self.pcds_cnt}")


    def get_best_pcd(self):
        points_counts = np.array([0] * self.pcds_cnt)
        noises = np.array([0] * self.pcds_cnt)
        densities = np.array([0] * self.pcds_cnt)

        for i in range(self.pcds_cnt):
            points_counts[i] = self.scores[i]["points_count"]
            noises[i] = self.scores[i]["noise"]
            densities[i] = self.scores[i]["density"]

        scaler = StandardScaler()
        points_counts = (scaler.fit_transform(points_counts.reshape(-1, 1))).flatten()
        noises = (scaler.fit_transform(noises.reshape(-1, 1))).flatten()
        densities = (scaler.fit_transform(densities.reshape(-1, 1))).flatten()

        self.tot_scores = [point_cnt - noise + density for point_cnt, noise, density in zip(points_counts, noises, densities)]
        max_score_idx = np.argmax(self.tot_scores)

        for i in range(self.pcds_cnt):
            print(f"pcd idx:{i} | tot score:{round(self.tot_scores[i], 5)}, points_count:{round(points_counts[i], 5)}, " +
                  f"noise:{round(noises[i], 5)}, density:{round(densities[i], 5)}")

        return self.pcds[max_score_idx], max_score_idx