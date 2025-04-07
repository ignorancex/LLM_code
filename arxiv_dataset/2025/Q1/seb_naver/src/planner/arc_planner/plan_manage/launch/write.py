#!/usr/bin/env python

def generate_launch_file(num_nodes):
    launch_file_content = '<launch>\n\n'

    for i in range(1, num_nodes + 1):
        node_name = "plan_node_{}".format(i)
        map_id = (i-1) * 1000 + 1

        node_content = '''
  <node pkg="plan_manage" type="plan_node" name="{node_name}" output="screen">
    <rosparam file="$(find plan_manage)/config/planning.yaml" command="load" />
    <param name="data/mapid" value="{map_id}" type="int"/>
    <param name="data/pathid" value="1" type="int"/>
  </node>\n
        '''.format(node_name=node_name, map_id=map_id)

        launch_file_content += node_content

    launch_file_content += '</launch>\n'

    return launch_file_content

if __name__ == '__main__':
    num_nodes = 15  # 设置节点数量
    
    launch_file_content = generate_launch_file(num_nodes)
    
    with open('/home/han/2023codes/NeuralTraj/src/plan_manage/launch/generated_launch.launch', 'w') as file:
        file.write(launch_file_content)
    
    print('Launch file "generated_launch.launch" generated successfully.')