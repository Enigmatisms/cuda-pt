import os

def list_obj_files(directory):
    # 遍历目标文件夹
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.obj'):
                # 去掉文件后缀并打印内容
                obj_name = file[:-4]
                print(f"<shape type=\"obj\">\n\t\t<string name=\"filename\" value=\"../meshes/kitchen/{file}\"/>\n\t\t<ref type=\"material\" id=\"white\"/>\n</shape>")

# 示例用法：指定要遍历的文件夹路径
if __name__ == "__main__":
    folder_path = "./kitchen"  # 替换为你的文件夹路径
    list_obj_files(folder_path)
