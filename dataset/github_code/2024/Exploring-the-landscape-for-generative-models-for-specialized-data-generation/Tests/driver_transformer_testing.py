from testing import Poly_Kernel_Testing


a = Poly_Kernel_Testing("/home/moe/Desktop/dof/output_with_rnn.csv","DOF_API/dataset (copy).csv",0.2,2)
b = a.inliers()
print(b)
a.draw(b)


a = Poly_Kernel_Testing("/home/moe/Desktop/dof/output_with_transformers.csv","DOF_API/dataset (copy).csv",0.2,2)
b = a.inliers()
print(b)
a.draw(b)
