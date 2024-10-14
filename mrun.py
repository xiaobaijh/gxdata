# import os
# import subprocess
# import glob

# # 定义根目录和detect.py脚本路径
# root_dir = "/home/wjh/code/yolov7/pictures/head"
# detect_script = "/home/wjh/code/yolov7/detect.py"


# # # 遍历根目录下的所有文件夹
# # for folder_name in os.listdir(root_dir):
# #     folder_path = os.path.join(root_dir, folder_name)
# #     if os.path.isdir(folder_path):
# #         # 构建命令
# #         command = f"python {detect_script} --source {folder_path}"
# #         # 运行命令
# #         subprocess.run(command, shell=True)
# jpg_count = 0
# for folder_name in os.listdir(root_dir):
#     folder_path = os.path.join(root_dir, folder_name)
#     jpgs = glob.glob(folder_path + '/*.jpg')
#     jpg_count += len(jpgs)

# print(jpg_count)

import os
import shutil


def move():
    # 定义源文件和目标文件夹
    source_file = "truck_new.txt"
    destination_folder = "/home/wjh/code/yolov7/pictures/truck_new"

    # 创建目标文件夹（如果不存在）
    os.makedirs(destination_folder, exist_ok=True)

    # 读取car.txt文件中的路径
    with open(source_file, "r") as file:
        lines = file.readlines()
    i = 0
    # 移动文件到新的文件夹
    for line in lines:
        file_path = line.strip()
        if os.path.isfile(file_path):
            shutil.copy(file_path, destination_folder)
            print(f"Moved: {file_path}")
            i += 1
        else:
            print(f"File not found: {file_path}")
    print(f"Total: {i}")


def check_labels():
    label_path = "/home/wjh/code/yolov7/data/labels"
    txt_path = "/home/wjh/code/yolov7/data/val.txt"
    with open(txt_path, "r") as file:
        lines = file.readlines()
    for line in lines:
        file_path = line.strip()
        label_name = file_path.split("/")[-1].replace("jpg", "txt")
        with open(os.path.join(label_path, label_name), "r") as file:
            label_lines = file.readlines()
        if len(label_lines) > 1:
            print(file_path)


def find():
    label_path = "/home/wjh/code/yolov7/data/labels"
    txt_path = "/home/wjh/code/yolov7/van.txt"

    with open(txt_path, "r") as file:
        lines = file.readlines()

    line_to_keep = []
    for line in lines:
        file = line.strip()
        label_name = file.split("/")[-1].replace("jpg", "txt")
        path = os.path.join(label_path, label_name)
        if os.path.exists(path):
            print(path)
        else:
            line_to_keep.append(line)

    with open("van_new.txt", "w") as file:
        for line in line_to_keep:
            file.write(line)


def detect():
    root = "/home/wjh/code/yolov7/pictures/head"
    basecommand = "python detect.py --save-txt ---source  "
    targets = ["car", "truck", "van"]
    for target in targets:
        for folder_name in os.listdir(root):
            folder_path = os.path.join(root, folder_name)
            if os.path.isdir(folder_path):
                command = basecommand + os.path.join(root, target)
                os.system(command)
                print(f"Finished {target} {folder_name}")


def mv_labels():
    label_path = "/home/wjh/code/yolov7/runs/detect/exp24/labels"
    pic_path = "/home/wjh/code/yolov7/runs/detect/exp24"
    for label in os.listdir(label_path):
        label_name = label.split(".")[0]
        pic_name = label_name + ".jpg"
        if not os.path.exists(os.path.join(pic_path, pic_name)):
            print(label_name)
            os.remove(os.path.join(label_path, label))



def find_and_copy():
    source_dir = "/home/wjh/code/yolov7/car_ano/2175477_1727152366/Images"
    target_base_dir = "/home/wjh/code/yolov7/pictures/head"
    output_dir = "/home/wjh/code/yolov7/data/images"


    # 获取日期范围内的所有日期目录
    date_dirs = [f"2023-02-{str(day).zfill(2)}" for day in range(4, 11)]

    # 遍历源目录下的所有文件
    for file_name in os.listdir(source_dir):
        source_file_path = os.path.join(source_dir, file_name)
        
        if os.path.isfile(source_file_path):
            # 在目标目录中查找文件
            file_found = False
            for date_dir in date_dirs:
                target_dir = os.path.join(target_base_dir, date_dir)
                target_file_path = os.path.join(target_dir, file_name)
                
                if os.path.exists(target_file_path):
                    # 如果文件存在，则复制到输出目录
                    shutil.copy(target_file_path, output_dir)
                    file_found = True
                    break
            
            if not file_found:
                print(f"文件 {file_name} 在目标目录中未找到。")

    print("文件复制完成。")


def remove_images_without_annotations():
    image_dir = "/home/wjh/code/yolov7/data/images"
    annotation_dir = "/home/wjh/code/yolov7/data/Annotations"

    # 遍历图像目录下的所有 JPG 文件
    for file_name in os.listdir(image_dir):
        if file_name.endswith('.jpg'):
            image_path = os.path.join(image_dir, file_name)
            xml_file_name = file_name.replace('.jpg', '.xml')
            xml_file_path = os.path.join(annotation_dir, xml_file_name)
            
            # 检查同名的 XML 文件是否存在
            if not os.path.exists(xml_file_path):
                print(f"删除图片: {image_path}")
                os.remove(image_path)                      
           
           
def count():
    import os

    # 定义类别标签
    labels = {0: 'truck', 1: 'van', 2: 'car', 3: 'people'}
    counts = {label: 0 for label in labels.values()}

    # 设置目录路径
    directory = '/home/wjh/code/yolov7/data/labels'

    # 遍历目录下的所有txt文件
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as file:
                for line in file:
                    class_id = int(line.split()[0])
                    if class_id in labels:
                        counts[labels[class_id]] += 1

    # 打印统计结果
    for label, count in counts.items():
        print(f'{label}: {count}')
    
                 
if __name__ == "__main__":
    count()
