import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, SetEntityState
import numpy as np
from itertools import product

class CustomObject():
    shapes = {"sphere":"<sphere><radius>0.5</radius></sphere>",
              "tetrahedron":"<mesh><uri>file:///home/tin/FER/Diplomski/4.semestar/Tetrahedron.stl</uri><scale>0.05 0.05 0.05</scale></mesh>"}
    
    colors = {"red":"Gazebo/Red",
              "green":"Gazebo/Green",
              "purple":"Gazebo/Purple",
              "blue":"Gazebo/Blue"}


#'<collision name="collision">'\
#   '<geometry>{}</geometry>'\
#</collision>'\
    #counter = np.zeros((3,3))
    xml_template = '<sdf version="1.6">'\
                    '<model name="{}">'\
                        '<pose>-5 0 0 0 0 0</pose>'\
                            '<link name="link">'\
                                '<gravity>false</gravity>'\
                                '<collision name="collision">'\
                                    '<geometry><sphere><radius>0.0</radius></sphere></geometry>'\
                                '</collision>'\
                                '<visual name="visual">'\
                                    '<geometry>{}</geometry>'\
                                    '<material>'\
                                        '<script>'\
                                            '<uri>file://media/materials/scripts/gazebo.material</uri>'\
                                            '<name>{}</name>'\
                                        '</script>'\
                                    '</material>'\
                                '</visual>'\
                            '</link>'\
                    '</model>'\
                    '</sdf>'

    def __init__(self, node, shape, color,t):
        self.node = node
        # choose random shape and color and increment counter
        #shape_index = np.random.randint(0,len(CustomObject.shapes))
        #color_index = np.random.randint(0,len(CustomObject.colors))
        self.shape = shape#list(CustomObject.shapes.keys())[shape_index]
        self.color = color#list(CustomObject.colors.keys())[color_index]
        #id_num = CustomObject.counter[shape_index,color_index]
        #CustomObject.counter[shape_index,color_index] += 1

        self.name = self.color + "_" + self.shape
        self.xml = CustomObject.xml_template.format(self.name, CustomObject.shapes[self.shape], CustomObject.colors[self.color])
        self.life = np.random.randint(3,8) # lifetime of 3 to 6 seconds
        #self.flag = False
        self.step = 0
        self.t = t
        self.direction = node.direction

        self.node.spawn_entity(self)
        self.timer = self.node.create_timer(node.timer_period, self.timer_callback)

    def timer_callback(self):
        self.timer.cancel()
        #if self.node.num_objects !=1: self.flag = True
        self.timer = self.node.create_timer(np.abs(self.t), self.timer_callback)
        self.life = np.random.randint(3,8)
        self.node.move_object(self)

        

class ObjectSpawner(Node):
    def __init__(self):
        #print("init")
        super().__init__('object_spawner')
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')
        self.velocity_client = self.create_client(SetEntityState, '/sim/set_entity_state')
        

        self.timer_period = 3  # seconds after spawn after which to start velocity
        self.objects = []
        self.direction = -1
        #self.num_objects = num_objects
        
        self.steps = 16
        self.width = 3.0
        self.increment = self.width/self.steps 
        row_t = 5
        
        obj_list = [("red","sphere",row_t)]#,("blue","sphere",-row_t)]

        for color, shape,t in obj_list:#product(CustomObject.colors, CustomObject.shapes):
            self.objects.append(CustomObject(self,shape,color,t))
        

    def spawn_entity(self, obj):
        # Spawn the object at a random position
        spawn_request = SpawnEntity.Request()
        spawn_request.name = obj.name
        spawn_request.xml = obj.xml
        self.spawn_client.call_async(spawn_request)

    def move_object(self, obj):
        # Move the object at a constant random velocity
        state = EntityState()
        state.name = obj.name

        # position
        state.pose.position.x = np.random.random(1)[0] * 3 + 1.5# [0.5, 4.5] # 4.0
        state.pose.position.y = np.random.random(1)[0] * 4 - 2 #- self.increment * int(((obj.step/obj.t)*self.steps**2%self.steps)) #np.random.random(1)[0] * 4 - 2 # [-2, 2] -self.direction * 3.0# np.sign(obj.t)*obj.direction * self.width/2
        state.pose.position.z = np.random.random(1)[0] *2#4.25 - self.increment * obj.step#int(((obj.step/obj.t)*self.steps**2//self.steps)) #np.random.random(1)[0] *2  # [1, 3] #1.0 + np.sign(obj.t)*0.5

        # orientation
        #q = np.random.randn(4)
        #q /= np.linalg.norm(q)
        #state.pose.orientation.x = q[0]
        #state.pose.orientation.y = q[1]
        #state.pose.orientation.z = q[2]
        #state.pose.orientation.w = q[3]

        # linear velocity
        v = np.random.randn(3) * 0.5 #6.0/self.timer_period
        state.twist.linear.x = 0.0#v[0] # maybe leave at 0.0, to keep movement on plane yz
        state.twist.linear.y = v[1]#obj.direction* -self.width/obj.t#v[1] #self.direction * v
        state.twist.linear.z = v[2]

        req = SetEntityState.Request()
        req._state = state
        self.velocity_client.call_async(req)
        obj.direction*=-1
        obj.step+=1
        obj.step%=self.steps

    def delete_object(self, obj):
        delete_request = DeleteEntity.Request()
        delete_request.name = obj.name
        self.delete_client.call_async(delete_request)
        self.objects.remove(obj)

        # create new object
        self.objects.append(CustomObject(self))

def main(args=None):
    rclpy.init(args=args)
    object_spawner = ObjectSpawner()
    rclpy.spin(object_spawner)

    rclpy.shutdown()
    print("shutdown")

if __name__ == '__main__':
    main()

