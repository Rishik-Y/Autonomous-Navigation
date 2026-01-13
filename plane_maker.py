import math

# Road parameters
straight_length = 20.0  # meters
road_width = 4.0        # meters
segment_length = 2.0    # small road segment length for circle approximation
circle_radius = 10.0    # radius of circular road

def write_box_road(name, size, origin_xyz, origin_rpy, file):
    file.write(f"""  <link name="{name}">
    <visual>
      <geometry>
        <box size="{size[0]} {size[1]} {size[2]}"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <box size="{size[0]} {size[1]} {size[2]}"/>
      </geometry>
    </collision>
  </link>
""")
    if origin_xyz is not None and origin_rpy is not None:
        parent = "road_straight_1" if name == "road_straight_2" else f"{name}_prev"
        file.write(f"""  <joint name="joint_{parent}_to_{name}" type="fixed">
    <parent link="{parent}"/>
    <child link="{name}"/>
    <origin xyz="{origin_xyz[0]:.3f} {origin_xyz[1]:.3f} {origin_xyz[2]:.3f}" rpy="{origin_rpy[0]:.3f} {origin_rpy[1]:.3f} {origin_rpy[2]:.3f}"/>
  </joint>
""")

with open("road_layout.urdf", "w") as f:
    f.write('<?xml version="1.0" ?>\n<robot name="road_layout">\n')

    # First straight road centered at (x=-10, y=0)
    f.write(f"""  <link name="road_straight_1">
    <visual>
      <geometry>
        <box size="{straight_length} {road_width} 0.1"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <box size="{straight_length} {road_width} 0.1"/>
      </geometry>
    </collision>
  </link>
""")

    # Second straight road (to connect after circle), no parent join needed
    f.write(f"""  <link name="road_straight_2">
    <visual>
      <geometry>
        <box size="{straight_length} {road_width} 0.1"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <box size="{straight_length} {road_width} 0.1"/>
      </geometry>
    </collision>
  </link>
""")

    # Joint from first straight to second straight with translation for connection via circle
    f.write(f"""  <joint name="joint_road_straight_1_to_road_straight_2" type="fixed">
    <parent link="road_straight_1"/>
    <child link="road_straight_2"/>
    <origin xyz="{straight_length*1.5:.3f} 0 0" rpy="0 0 0"/>
  </joint>
""")

    # Approximate the circular road with multiple small segments in semicircle (180 deg)
    circumference = math.pi * circle_radius * 2
    arc_length = math.pi * circle_radius  # semicircle length
    num_segments = int(arc_length / segment_length)

    # Starting angle for segments at 0 degrees on circle (right side)
    start_angle = 0.0
    prev_link = "road_straight_1"
    for i in range(num_segments):
        angle = start_angle + (i + 0.5) * (math.pi / num_segments)  # segment center angle

        # Position of segment center along circumference
        x = circle_radius * math.cos(math.pi/2 - angle)
        y = circle_radius * math.sin(math.pi/2 - angle)
        z = 0

        # Orientation: box rotated to be tangent to circle (around z-axis)
        yaw = -angle + math.pi / 2  # tangent angle

        link_name = f"road_circle_seg_{i+1}"
        size = (segment_length, road_width, 0.1)

        # Write road segment
        f.write(f"""  <link name="{link_name}">
    <visual>
      <geometry>
        <box size="{size[0]} {size[1]} {size[2]}"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <box size="{size[0]} {size[1]} {size[2]}"/>
      </geometry>
    </collision>
  </link>
  <joint name="joint_{prev_link}_to_{link_name}" type="fixed">
    <parent link="{prev_link}"/>
    <child link="{link_name}"/>
    <origin xyz="{x:.3f} {y:.3f} {z:.3f}" rpy="0 0 {yaw:.3f}"/>
  </joint>
""")

        prev_link = link_name

    # Connect last circle segment to second straight road
    f.write(f"""  <joint name="joint_{prev_link}_to_road_straight_2" type="fixed">
    <parent link="{prev_link}"/>
    <child link="road_straight_2"/>
    <origin xyz="{straight_length*2:.3f} 0 0" rpy="0 0 0"/>
  </joint>
""")

    f.write("</robot>\n")

print("road_layout.urdf successfully generated with straight and semicircular roads.")
