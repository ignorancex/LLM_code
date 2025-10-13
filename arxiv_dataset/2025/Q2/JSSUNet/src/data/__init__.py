batch_shape_dict = {
                    "sssr": lambda gt, rgb, low_hs, low, clusters, name: (low, {"rgb": rgb, "low_hs": low_hs, "gt": gt}, name),
                    "sssr_kmeans": lambda gt, rgb, low_hs, low, clusters, name: ((low, clusters), {"rgb": rgb, "low_hs": low_hs, "gt": gt}, name),
                    "sr_hs": lambda gt, rgb, low_hs, low, clusters, name: (low_hs, {"gt": gt}, name),
                    "ssr": lambda gt, rgb, low_hs, low, clusters, name: (rgb, {"gt": gt}, name),
                    "ssr_low": lambda gt, rgb, low_hs, low, clusters, name: (low, {"gt": low_hs}, name),
                    "ssr_kmeans": lambda gt, rgb, low_hs, low, clusters, name: ((rgb,clusters), {"gt": gt}, name),
                    "ssr_kmeans_low": lambda gt, rgb, low_hs, low, clusters, name: ((low, clusters), {"gt": low_hs}, name),
                    "sr_rgb": lambda gt, rgb, low_hs, low, clusters, name: (low, {"gt": rgb}, name),
                    "fusion": lambda gt, rgb, low_hs, low, clusters, name: ((low_hs, rgb), {"gt": gt}, name)
                    }