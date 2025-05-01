import os
import glob
import cv2
import numpy as np
import sys
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# 디렉토리 경로 가져오기
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# 상위 디렉토리에 결과물을 저장할 폴더 생성
output_folder_name = "upscaled_thermal_images3"
parent_directory = os.path.dirname(script_directory)
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
    image_files.extend(glob.glob(os.path.join(script_directory, f"*{ext}")))
    image_files.extend(glob.glob(os.path.join(script_directory, f"*{ext.upper()}")))

print(f"발견된 이미지 파일 수: {len(image_files)}")

if len(image_files) == 0:
    print("처리할 이미지 파일이 없습니다. 프로그램을 종료합니다.")
    sys.exit(0)

# 모델 파일 경로 설정
model_path = os.path.join(script_directory, 'RealESRGAN_x4plus.pth')

# 모델 파일이 없으면 다운로드
if not os.path.exists(model_path):
    print("Real-ESRGAN 모델 다운로드 중...")
    import requests
    
    url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"모델 다운로드 완료: {model_path}")
    else:
        print("모델 다운로드 실패. 수동으로 다운로드하세요.")
        print("URL: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")
        sys.exit(1)

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 장치: {device}")

try:
    # RealESRGAN 모델 설정
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
    
    # 업샘플러 초기화
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True if torch.cuda.is_available() else False
    )
    
    print("Real-ESRGAN 모델 로드 완료")
    
    # 업스케일링 수행
    for idx, image_path in enumerate(image_files, 1):
        try:
            # 파일 이름 추출
            file_name = os.path.basename(image_path)
            file_base, file_ext = os.path.splitext(file_name)
            
            print(f"[{idx}/{len(image_files)}] 처리 중: {file_name}")
            
            img = cv2.imread(image_path)
            if img is None:
                print(f"이미지를 읽을 수 없습니다: {image_path}")
                continue
            
            # 원본 이미지 크기 저장
            original_height, original_width = img.shape[:2]
            
            # 노이즈 제거 / Real-ESRGAN으로 업스케일링
            img_denoised = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
            output, _ = upsampler.enhance(img_denoised, outscale=4)
            
            output_path = os.path.join(output_directory, f"{file_base}_upscaled{file_ext}")
            cv2.imwrite(output_path, output)

            print(f"  - 원본 크기: {original_width}x{original_height}")
            print(f"  - 업스케일링 후 크기: {output.shape[1]}x{output.shape[0]}")
            print(f"  - 저장 완료: {output_path}")
            
        except Exception as e:
            print(f"이미지 처리 중 오류 발생: {image_path}")
            print(f"오류 내용: {str(e)}")
    
    print(f"\n모든 이미지 처리 완료. 결과물 저장 위치: {output_directory}")

except Exception as e:
    print(f"모델 로드 또는 처리 중 오류 발생: {str(e)}")
    print("pip install basicsr facexlib gfpgan realesrgan opencv-python torch torchvision")
