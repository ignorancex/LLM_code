import os
import sys

A_val=0.3
B_val=0.5
M_val=float(sys.argv[1])
Floops=7

for Lx in [4,6]:
    for Ly in [4]:
        out1 = f"input_BHZ_{Lx}x{Ly}.ain"

        if os.path.exists(out1):
            os.remove(out1)
        else:
            None

        with open(out1, "w") as f:

            f.write("##Ainur1.0\n")
            f.write(f"#{Lx}x{Ly} ladder\n\n")

            Sites = Lx*Ly

            f.write(f"TotalNumberOfSites={Sites};\n")
            f.write("NumberOfTerms=1;\n\n")

            f.write("Model=\"FeAsBasedScSpinDependentHopping\";\n")
            f.write("FeAsMode=\"INT_PAPER33\";\n")
            f.write("Orbitals=2;\n")
            f.write("SolverOptions=\"twositedmrg,useComplex\";\n")
            f.write("Version=\"6.01\";\n")
            f.write("OutputFile=\"data324\";\n")
            f.write("InfiniteLoopKeptStates=200;\n")

            state_arr = [(sval * 300) + 200 for sval in range(1, Floops - 1)]
            state_arr.append((Floops - 2) * 300 + 500)
            state_arr.append((Floops - 2) * 300 + 500)

            f.write("FiniteLoops=[\n")
            for num in range(1, Floops):
                f.write(f"[@auto, {state_arr[num - 1]}, 0],\n")
            f.write(f"[@auto, {state_arr[Floops - 2]}, 1]\n")
            f.write("];\n\n")

            f.write(f"TargetElectronsUp={Sites};\n")
            f.write(f"TargetElectronsDown={Sites};\n")
            f.write("Threads=npthreads_val;\n\n")

            f.write("#There are 4 values of U: [U,U'-J/2,-2J,-J]\n")
            f.write("hubbardU=[U_val,Up_m_Jby2_val,m_2J_val,m_J_val];\n\n")

            Mm4B = 1.0*M_val

            f.write("#Inserting onsite-potential V: site->orb->spin\n")

            Varr_up = []
            Varr_dn = []
            for orb in range(2):
                for site in range(Sites):
                    ind = site + Sites * orb
                    Varr_up.append(Mm4B * (-1.0) ** (orb+1))
                    Varr_dn.append(Mm4B * (-1.0) ** (orb+1))

            doubleSites = 2*Sites
            f.write("potentialV=[\n")
            for inup in range(doubleSites):
                f.write(f"{Varr_up[inup]},")
            f.write("\n")
            for indn in range(doubleSites - 1):
                f.write(f"{Varr_dn[indn]},")
            f.write(f"{Varr_dn[doubleSites - 1]}")
            f.write("\n];\n\n")

            f.write("#Inserting hopping connections: orb->spin->site\n")
            f.write("DegreesOfFreedom=4;\n")
            f.write("GeometryKind=\"LongRange\";\n")
            f.write("GeometryOptions=\"none\";\n")
            f.write(f"integer GeometryMaxConnections={Ly+1};\n")

            f.write("matrix Connectors=[\n")

            os.system(f"./long_range_connections_exe {Lx} {Ly} {A_val} {B_val} out.txt")

            with open(f"out.txt", "r") as out_file:
                f.write(out_file.read())
            f.write("];\n")
