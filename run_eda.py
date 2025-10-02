import os
import subprocess

# 처리할 상위 폴더와 출력 폴더
folders = [
    # "sample/lucchi",
    "sample/nuclei",
    # "sample/brats",
    # "sample/covid",
    # "sample/synthrad"
]

output_base = "outputs"

# 확장자 필터
img_exts = (".png", ".tif", ".jpg", ".jpeg")

for folder in folders:
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(img_exts):
                input_path = os.path.join(root, f)
                
                # 상대 경로에서 sample/ 뒤 경로 추출
                rel_path = os.path.relpath(input_path, "sample")
                # 확장자 제거
                rel_path_noext = os.path.splitext(rel_path)[0]
                output_path = os.path.join(output_base, rel_path_noext)

                # 폴더 자동 생성
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # eda.py 실행
                cmd = [
                    "python", "eda.py",
                    f"image.image.image_path={input_path}",
                    f"image.image.output_path={output_path}"
                ]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd)
