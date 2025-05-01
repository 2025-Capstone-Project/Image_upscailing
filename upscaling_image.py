import os
import glob
import cv2
from pathlib import Path
import numpy as np

# 현재 디렉토리
current_directory = os.path.dirname(os.path.abspath(__file__))
print(f"현재 작업 디렉토리: {current_directory}")

# 상위 디렉토리에 결과물을 저장할 폴더 생성
output_folder_name = "upscaled_thermal_images2"
parent_directory = os.path.dirname(current_directory)
output_directory = os.path.join(parent_directory, output_folder_name)

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"결과물 저장 폴더 생성 완료: {output_directory}")
else:
    print(f"이미 결과물 저장 폴더가 존재합니다: {output_directory}")

# 이미지 파일 찾기
image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
image_files = []

for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(current_directory, f"*{ext}")))
    image_files.extend(glob.glob(os.path.join(current_directory, f"*{ext.upper()}")))

print(f"발견된 이미지 파일 수: {len(image_files)}")

# 업스케일링 함수 정의 (초해상화)
def upscale_image(image_path, scale_factor=2):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        return None
    
    # 업스케일링 INTER_CUBIC 보간법 사용
    upscaled_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    return upscaled_img

# 업스케일링 수행
for image_path in image_files:
    try:
        file_name = os.path.basename(image_path)
        file_base, file_ext = os.path.splitext(file_name)
        
        # 업스케일링 수행 (스케일 팩터 4)
        upscaled_img = upscale_image(image_path, scale_factor=4)
        
        if upscaled_img is not None:
            # 결과 저장
            output_path = os.path.join(output_directory, f"{file_base}_upscaled{file_ext}")
            cv2.imwrite(output_path, upscaled_img)
            
            original_img = cv2.imread(image_path)
            print(f"처리 완료: {file_name}")
            print(f"  - 원본 크기: {original_img.shape[1]}x{original_img.shape[0]}")
            print(f"  - 업스케일링 후 크기: {upscaled_img.shape[1]}x{upscaled_img.shape[0]}")
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {image_path}")
        print(f"오류 내용: {str(e)}")

print(f"\n모든 이미지 처리 완료. 결과물 저장 위치: {output_directory}")
