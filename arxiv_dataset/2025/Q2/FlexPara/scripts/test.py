import sys,os
sys.path.append(os.path.abspath(".."))
from cdbs.funcs import *
from cdbs.workflow import * 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("load_mesh_path", type=str)
    parser.add_argument("load_ckpt_path", type=str)
    parser.add_argument("export_root", type=str)
    args = parser.parse_args()
    
    mesh_name = args.load_mesh_path.split("/")[-1].split(".")[0]
    if args.load_ckpt_path == "flexpara_global.pth":
        export_folder = os.path.join(args.export_root, mesh_name+"_1")
        charts_number = 1
    else:
        charts_number = int(args.load_ckpt_path.replace("flexpara_multi_", "").replace(".pth", ""))
        export_folder = os.path.join(args.export_root, mesh_name+"_"+str(charts_number))

    vertices, normals, faces = load_mesh_model_vfn(args.load_mesh_path)
    num_verts = vertices.shape[0]
    num_faces = faces.shape[0]
    print(f"{num_verts=}, {num_faces=}")
    
    print(f"start testing on [{mesh_name}] ...")
    test_flexpara(vertices, normals, faces, args.load_ckpt_path, export_folder, charts_number)

if __name__ == '__main__':
    main()
    print("testing finished.")