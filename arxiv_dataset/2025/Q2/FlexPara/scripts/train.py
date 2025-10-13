import sys,os
sys.path.append(os.path.abspath(".."))
from cdbs.funcs import *
from cdbs.workflow import * 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("charts_number", type=int, help="if charts_number = 1, do global parameterization")
    parser.add_argument("load_mesh_path", type=str)
    parser.add_argument("export_root", type=str)
    parser.add_argument("N", type=int, help="number of points fed into FAM during each iteration",default=1600)
    parser.add_argument("num_iter", type=int)
    args = parser.parse_args()
    
    mesh_name = args.load_mesh_path.split("/")[-1].split(".")[0]
    export_folder = os.path.join(args.export_root, mesh_name+"_"+str(args.charts_number))
    os.makedirs(export_folder, exist_ok=True)
    
    vertices, normals, faces = load_mesh_model_vfn(args.load_mesh_path)
    num_verts = vertices.shape[0]
    num_faces = faces.shape[0]
    print(f"{num_verts=}, {num_faces=}")
    
    print(f"start training on [{mesh_name}] ...")
    train_flexpara(vertices, normals, faces, args.charts_number ,args.num_iter, export_folder,args.N)

if __name__ == '__main__':
    main()
    print("training finished.")