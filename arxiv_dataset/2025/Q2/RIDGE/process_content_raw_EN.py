import glob
import os
import shutil
import pandas as pd
import argparse


def main(args):
    content_dir = os.path.join("contents", args.content_dir)
    filenames = sorted(glob.glob(f'{content_dir}/*.txt'))
    # print(filenames)

    num_in_h_error = 0
    blacklist = [] # (error message, filename)

    for filename in filenames:
        with open(filename, 'r', encoding="utf8") as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                # remove tab (2 spaces)
                if line.startswith('  ') and line.replace(' ', '').strip() != '':
                    idx = -1
                    for char in line:
                        if ord(char) == 32:
                            idx += 1
                        else:
                            break
                    if idx != -1:
                        num_dash = (idx + 1) // 2
                        line = "-" * num_dash + line[num_dash*2:]

                # remove '\n' at the end of each line
                line = line.strip()

                # replace strange icon
                line = line.replace('⚫', '☑').replace('⭕', '☐')
                line = line.replace('✔️', '☑').replace('❌', '☐')

                lines[i] = line # truely update line in lines

            # check and add </> at the end of each tag section
            new_lines = lines.copy()
            in_h = False
            cur_h_num = 0
            for line in lines:
                
                if line.startswith('<h'):
                    cur_h_num = int(line.split('>')[0][2:])
                    # print(cur_h_num)
                    if in_h:
                        last_exist_text_idx = new_lines.index(line) - 1
                        while new_lines[last_exist_text_idx] == "":
                            last_exist_text_idx -= 1
                        new_lines.insert(last_exist_text_idx+1, f"</h{cur_h_num-1}>")
                    in_h = True

                elif line.startswith('</h'):
                    in_h = False

                elif line.startswith("</content>"):
                    if in_h:
                        last_exist_text_idx = new_lines.index(line) - 1
                        while new_lines[last_exist_text_idx] == "":
                            last_exist_text_idx -= 1
                        new_lines.insert(last_exist_text_idx+1, f"</h{cur_h_num}>")
            
            lines = new_lines.copy()

            # check again
            in_h = False
            in_h_error = False
            for i, line in enumerate(lines):
                if line.startswith('<h'):
                    if in_h:
                        # print("\n".join(lines))
                        print(filename)
                        blacklist.append((f"Error in </> line {line}", filename))
                        num_in_h_error += 1
                        in_h_error = True
                        break
                    else:
                        in_h = True
                elif line.startswith('</h'):
                    if not in_h:
                        # print("\n".join(lines))
                        print(filename)
                        blacklist.append((f"Error in </> line {line}", filename))
                        num_in_h_error += 1
                        in_h_error = True
                        break
                    else:
                        in_h = False
                elif line.startswith("</content>"):
                    if in_h:
                        # print("\n".join(lines))
                        print(filename)
                        blacklist.append((f"Error </> before </content> line", filename))
                        num_in_h_error += 1
                        in_h_error = True
                        break
            
            if in_h_error:
                continue


            # make "<content>" and "</content>" in a single line
            for line in lines:
                if line.find("<content>") != -1 and line != "<content>":
                    print(filename, 1)
                    lines[lines.index(line)] = "<content>"
                    lines.insert(lines.index("<content>")+1, line.replace("<content>", "").strip())
                if line.find("</content>") != -1 and line != "</content>":
                    print(filename, 2)
                    lines[lines.index(line)] = "</content>"
                    lines.insert(lines.index("</content>"), line.replace("</content>", "").strip())
            

            # check "-" level
            stack = []

            flag = False # ending tag
            error_flag = False
            in_h_flag = False
            for i, line in enumerate(lines[1:]): # exclude title
                if flag:
                    print(f"ERROR </content> in {filename}: {line}")
                    blacklist.append((f"ERROR </content>", filename))
                    error_flag = True
                    # exit()
                    continue

                if i == 0 and line != "<content>":
                    print(f"ERROR <content> in {filename}: {line}")
                    blacklist.append((f"ERROR <content>", filename))
                    error_flag = True
                    # exit()
                    continue
                if i == len(lines) - 2 and line != "</content>":
                    print(f"ERROR </content> in {filename}: {line}")
                    blacklist.append((f"ERROR </content>", filename))
                    error_flag = True
                    # exit()
                    continue

                if line == "<content>" or line == "":
                    continue
                elif line == "</content>":
                    flag = True
                    continue

                if not line.startswith("<h") and not line.startswith("</h") and not line.startswith("-") :
                    print(f"ERROR in {filename} {i+2}: {line}")
                    blacklist.append((f"ERROR start format", f"{filename} {i+2}"))
                    error_flag = True
                    # exit()
                    continue

                # start checking "-" level
                cur_status = 0
                while(line.startswith("-")):
                    cur_status += 1
                    line = line[1:]
                line = line.strip()
                if line == "":
                    print(f"ERROR in {filename} {i+2}: No content in line {i+2}, {lines[i+1]}")
                    # print(f"====\n{lines}\n====")
                    blacklist.append((f"ERROR no content", f"{filename} {i+2}"))
                    error_flag = True
                    # exit()
                    continue

                if line.startswith("<h"):
                    stack = [0]
                    in_h_flag = True
                elif line.startswith("</h"):
                    stack = []
                    in_h_flag = False
                elif len(stack) > 0:
                    if cur_status - stack[-1] > 1:
                        print(f"ERROR in {filename} {i+2} -> {line}")
                        print("Cur status:", cur_status)
                        print("Pre status:", stack[-1])
                        blacklist.append((f"ERROR in layer", f"{filename} {i+2}"))
                        error_flag = True
                        # exit()
                        continue
                    elif cur_status - stack[-1] == 1:
                        stack.append(cur_status)
                    elif cur_status - stack[-1] == 0:
                        stack.append(cur_status)
                    elif cur_status - stack[-1] < 0:
                        while stack[-1] > cur_status:
                            stack.pop()
                        stack.append(cur_status)
                
                # stack being empty
                elif len(stack) == 0:
                    if in_h_flag:
                        print(f"ERROR in {filename} {i+2} -> {line}")
                        blacklist.append((f"ERROR no header", f"{filename} {i+2}"))
                        error_flag = True
                        # exit()
                        continue
                    else:
                        stack.append(cur_status)

                # print(stack)
            
            f.close()

        if error_flag:
            continue

        # write to file
        with open(filename, 'w', encoding="utf8") as f:
            f.write("\n".join(lines))
            f.close()

    print(f"Number of </> error: {num_in_h_error}")

    # deal with blacklist (move them to another folder)
    blacklist_dir = f"{content_dir}/blacklist"
    if not os.path.exists(blacklist_dir):
        os.makedirs(blacklist_dir)
    else:
        shutil.rmtree(blacklist_dir)
        os.makedirs(blacklist_dir)

    df = pd.DataFrame(blacklist, columns=["error_message", "filename (line)"])
    df.to_csv(f"{blacklist_dir}/blacklist.csv", index=False)

    # copy files to blacklist dir
    for error_message, filename in blacklist:
        if filename.find(" ") != -1:
            filename = filename.split(" ")[0]
        shutil.copy(filename, f"{blacklist_dir}/{os.path.basename(filename)}")

    num_blacklist_file = len(glob.glob(f"{blacklist_dir}/*.txt"))
    print("Num blacklist:", num_blacklist_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", type=str, default="example")
    args = parser.parse_args()

    main(args)