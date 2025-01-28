import sys
import os

X = "../meshes/rich/vision"

def generate_xml_elements(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not valid")
        return

    num_obj_file = 0
    for filename in os.listdir(directory):
        if filename.lower().endswith('.obj') and os.path.isfile(os.path.join(directory, filename)):
            num_obj_file += 1
            file_path = os.path.join(X, filename).replace(os.sep, '/')
            
            print(f'''<shape type="obj">
    <string name="filename" value="{file_path}"/>
    <ref type="material" id="white"/>
</shape>''')
    print(f"\nNumber of file: {num_obj_file}")
        

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ./shape_auto_read.py <path to the folder that contains the OBJ files>")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    generate_xml_elements(target_dir)