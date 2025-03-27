import os
import torch
import numpy as np
from PIL import Image
from .utils import visualize


class MapConverter:
    def __init__(
        self,
        h_polar=256,
        w_polar=704,
        h_cart=200,
        w_cart=200,
        extents_polar=[0, 53],
        extents_cart=[-25, 0, 25, 50],
        cam2img=[[1266, 0, 816]],
        image_width=1600,
        colormap=None,
        device=None,
    ):
        t1 = np.arctan(-cam2img[0][2] / cam2img[0][0])
        t2 = np.arctan((image_width - cam2img[0][2]) / cam2img[0][0])
        r1, r2 = extents_polar
        x1, z1, x2, z2 = extents_cart

        t = np.arange(t1, t2, (t2 - t1) / w_polar)[None, :]
        r = np.arange(r1, r2, (r2 - r1) / h_polar)[:, None]
        c2p_coord = np.stack([np.cos(t) * r, np.sin(t) * r], axis=0)
        c2p_coord[0] = (c2p_coord[0] - z1) / (z2 - z1) * h_cart
        c2p_coord[1] = (c2p_coord[1] - x1) / (x2 - x1) * w_cart
        c2p_index = np.around(c2p_coord).astype(np.int64)
        c2p_index[0] = np.clip(c2p_index[0], a_min=0, a_max=h_cart - 1)
        c2p_index[1] = np.clip(c2p_index[1], a_min=0, a_max=w_cart - 1)

        x = np.arange(x1, x2, (x2 - x1) / w_cart)[None, :]
        z = np.arange(z1, z2, (z2 - z1) / h_cart)[:, None]
        z[0] += 0.1
        p2c_coord = np.stack([np.sqrt(x**2 + z**2), np.arctan(x / z)], axis=0)
        p2c_coord[0] = (p2c_coord[0] - r1) / (r2 - r1) * h_polar
        p2c_coord[1] = (p2c_coord[1] - t1) / (t2 - t1) * w_polar
        p2c_index = np.around(p2c_coord).astype(np.int64)
        p2c_mask = ((p2c_index[1] < 0) | (p2c_index[1] >= w_polar))[None, :, :]
        p2c_index[0] = np.clip(p2c_index[0], a_min=0, a_max=h_polar - 1)
        p2c_index[1] = np.clip(p2c_index[1], a_min=0, a_max=w_polar - 1)

        self.c2p_index = torch.tensor(c2p_index, device=device)
        self.p2c_index = torch.tensor(p2c_index, device=device)
        self.p2c_mask = torch.tensor(p2c_mask, device=device)

        self.colormap = colormap
        if self.colormap is None:
            self.colormap = {
                "drivable_area": (166, 206, 227),  # '#a6cee3'
                "ped_crossing": (251, 154, 153),  # '#fb9a99'
                "walkway": (227, 26, 28),  # '#e31a1c'
                "carpark_area": (255, 127, 0),  # '#ff7f00'
                "car": (255, 158, 0),  # Orange
                "truck": (255, 99, 71),  # Tomato
                "trailer": (255, 140, 0),  # Darkorange
                "bus": (255, 127, 80),  # Coral
                "construction_vehicle": (233, 150, 70),  # Darksalmon
                "bicycle": (220, 20, 60),  # Crimson
                "motorcycle": (255, 61, 99),  # Red
                "pedestrian": (0, 0, 230),  # Blue
                "traffic_cone": (47, 79, 79),  # Darkslategrey
                "barrier": (112, 128, 144),  # Slategrey
                "mask": (63, 63, 63),  # Grey
            }  # RGB

    def cart2polar(self, map_cart):
        """Convert map from cart to polar.

        Args:
            map_cart (torch.bool): (*, 200, 200)

        Returns:
            map_polar (torch.bool): (*, 256, 352)
        """
        return map_cart[:, self.c2p_index[0], self.c2p_index[1]]

    def polar2cart(self, map_polar):
        """Convert map from polar to cart.

        Args:
            map_polar (torch.float): (*, 256, 352)

        Returns:
            map_cart (torch.float): (*, 200, 200)
        """
        return map_polar[:, self.p2c_index[0], self.p2c_index[1]] * ~self.p2c_mask


if __name__ == "__main__":
    """
    from nuscenes import NuScenes

    nuscenes = NuScenes("v1.0-trainval", "data/nuscenes")
    sample_data_examples = {}
    i = 0
    for sample in nuscenes.sample:
        if i % 10 == 0:
            sample_data_token = sample["data"]["CAM_FRONT"]
            sample_data = nuscenes.get("sample_data", sample_data_token)
            sample_data_examples[sample_data_token] = sample_data["filename"]
        i += 1
        if i >= 500:
            break
    """

    sample_data_examples = {
        "020d7b4f858147558106c504f7f31bef": "samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883530412470.jpg",
        "f1826933e09349c0bb5606bd3555cb43": "samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883535412467.jpg",
        "a59c061fc9b7488693b91d0bab13acf1": "samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883540512466.jpg",
        "84617f2bb3c14d5ca88c85e8134bf355": "samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883545512466.jpg",
        "8ed2a7ea9a894eac8b3cd43b706f39bc": "samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883719412465.jpg",
        "53b046512b9c4bf4817ead9de1003e5b": "samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883724412465.jpg",
        "fd70ace34db34ec6a7876165bf760c05": "samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883729362460.jpg",
        "5ce38f21b9904901a7b4fd49e81d66a5": "samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883734512464.jpg",
        "b8fba7d78cf547b996c431dec1f5ee26": "samples/CAM_FRONT/n015-2018-08-02-17-16-37+0800__CAM_FRONT__1533201470412460.jpg",
        "250018ac46314ca4873919f3cde82a8c": "samples/CAM_FRONT/n015-2018-08-02-17-16-37+0800__CAM_FRONT__1533201475412460.jpg",
        "7092214a72e5418c8d04044f134f7ef7": "samples/CAM_FRONT/n015-2018-08-02-17-16-37+0800__CAM_FRONT__1533201480412460.jpg",
        "0931140a78f04c77916259e85a57869a": "samples/CAM_FRONT/n015-2018-08-02-17-16-37+0800__CAM_FRONT__1533201485412460.jpg",
        "02e1485c2ca9423cb8505ddba23f6594": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531883985912463.jpg",
        "59ebe885635d46008b533e3ee3b27680": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531883990862460.jpg",
        "921988c7f51041209407265a7ab86ad2": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531883995912466.jpg",
        "5c95570e685d448f8b4dfc895c13b14a": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884000762460.jpg",
        "bdf062f77bbe4f07b6332df1d1acb190": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884104112478.jpg",
        "977f8571a8414c6aaee96db965974d1c": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884109012467.jpg",
        "436dd3fe276d4dd9b0ca4f25aee04f28": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884113912463.jpg",
        "81ffc4b369ed48248b4f2f45ff8352d3": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884118912464.jpg",
        "d0f5dabbd4cb41d4b07c58f5e0b20638": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884157412465.jpg",
        "751ed4dd638b4795a4ea7b3a3073b477": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884162412464.jpg",
        "7c62a19d707a45de984b75d7ba04f599": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884167412468.jpg",
        "a975382224254efea60fb56efe2224b5": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884172362460.jpg",
        "85a03d92fc154a9a86cdbdb7dc753101": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884258512467.jpg",
        "66635de1d36d46ab956c186e3876f2ee": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884263662460.jpg",
        "eeaae10aac0f4c13bae327f15e7ff662": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884268612461.jpg",
        "fb79fa3f68ca4c8ca2880e385cdf9f45": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884273662460.jpg",
        "4e7da406c02e47deb8fc63c87644df72": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884278912461.jpg",
        "83be1bea4afd4f9e81ad8756d80528c2": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884283912464.jpg",
        "dd6531413c6e44d496c0d9e2a124f493": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884288662460.jpg",
        "ab4017c0df0f45038f2c5a2e6edf6416": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884293612469.jpg",
        "0257d7d7070d4645806f8f1f75ad2175": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884298862460.jpg",
        "16520273ee504e1f8cacbb113a0960cb": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884303762460.jpg",
        "78a5c68f696c4f7fa760e5104d0313b3": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884308762460.jpg",
        "5bd41baa9bb54ccc884d37901d00e0ea": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884313662460.jpg",
        "a82a87cf827c49a8b54b925d02618df4": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884392012469.jpg",
        "7ad8298c3e1b47749b1e93c2fdc3b39f": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884397012461.jpg",
        "5f5f4ba3f3a6446db400a61a72e85467": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884402012464.jpg",
        "4f299ffc8d1743f9ab24a9470f6ec518": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884406912464.jpg",
        "a8fd875028014002919c75558fbfd69f": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884526912468.jpg",
        "7130d9e7be4c4e4dbd80bef227b02bc3": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884531912467.jpg",
        "92adf67eee0d434eb60f973cbde1c940": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884536862460.jpg",
        "0f7b29a747624982ac8642160b96334a": "samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884541862460.jpg",
        "de54a187aa4c49f7983913354bbbe159": "samples/CAM_FRONT/n015-2018-07-18-11-41-49+0800__CAM_FRONT__1531885321012467.jpg",
        "9bc566ecf6ce42cbb3dde1ab5ec1880b": "samples/CAM_FRONT/n015-2018-07-18-11-41-49+0800__CAM_FRONT__1531885325912470.jpg",
        "9bb7644031204f6c902faa96c88e21fa": "samples/CAM_FRONT/n015-2018-07-18-11-41-49+0800__CAM_FRONT__1531885330912464.jpg",
        "e924cf300e1647e5853945479e18645e": "samples/CAM_FRONT/n015-2018-07-18-11-41-49+0800__CAM_FRONT__1531885335912464.jpg",
        "76f58c43b338469c903b6fdeb8473e57": "samples/CAM_FRONT/n015-2018-07-18-11-41-49+0800__CAM_FRONT__1531885341012469.jpg",
        "b66795f6267940cfa3bd355f45b151ca": "samples/CAM_FRONT/n015-2018-07-18-11-41-49+0800__CAM_FRONT__1531885345912465.jpg",
    }

    map_converter = MapConverter(device="cuda")
    for token, image_filename in sample_data_examples.items():
        bev_map = np.load(os.path.join("data/nuscenes/bev_maps/CAM_FRONT", token + ".npz"))["bev_map"]
        bev_map = torch.tensor(bev_map, device="cuda")

        expand_map = map_converter.cart2polar(bev_map)
        expand_map = expand_map.to(torch.float32)
        gen_map = map_converter.polar2cart(expand_map)

        out_dir = os.path.join("outputs/map_convert_examples", token)
        image = Image.open(os.path.join("data/nuscenes", image_filename))
        visualize(
            out_dir=out_dir,
            image=image,
            bev_map=bev_map,
            expand_map=expand_map == 1,
            gen_map=gen_map == 1,
        )
