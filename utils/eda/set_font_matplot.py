import os
import zipfile
import subprocess
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def install_nanumgothic():
    font_dir = "./fonts"
    font_path = os.path.join(font_dir, "NanumGothicCoding.ttf")
    font_zip_path = os.path.join(font_dir, "NanumGothicCoding-2.5.zip")

    if not os.path.exists(font_dir):
        os.makedirs(font_dir)

    if not os.path.exists(font_path):
        print("NanumGothic 폰트를 설치 중입니다...")

        if not os.path.exists(font_zip_path):
            subprocess.run(["wget", "https://github.com/naver/nanumfont/releases/download/VER2.5/NanumGothicCoding-2.5.zip", "-P", font_dir], check=True)
        
        try:
            with zipfile.ZipFile(font_zip_path, 'r') as zip_ref:
                zip_ref.extractall(font_dir)
            print("폰트 압축 해제 완료")
        except zipfile.BadZipFile as e:
            print(f"ZIP 파일 해제 실패: {e}")
            return

        print("NanumGothic 폰트 설치 완료")


def set_nanumgothic_font():
    font_dir = "./fonts"
    font_path = font_dir + "/NanumGothicCoding.ttf"
    
    if not os.path.exists(font_dir):
        os.makedirs(font_dir)

    if not os.path.exists(font_path):
        install_nanumgothic()
    
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)

    plt.rcParams['font.family'] = font_prop.get_name()

    print("한글 폰트 설정 완료")


def list_available_fonts():
    """
    시스템에 설치된 폰트 목록을 출력하는 함수.
    """
    fonts = [f.name for f in fm.fontManager.ttflist]
    
    print("사용 가능한 폰트 목록:")

    for font in sorted(fonts):
        print(font)

    print("NanumGothicCoding" in fonts)