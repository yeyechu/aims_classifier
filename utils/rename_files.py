import os


def rename_files_in_folder(folder_path, rename_rule, start_index=1):
    """
    폴더 내 파일 이름을 변경하는 함수.

    Args:
        folder_path (str): 파일이 있는 폴더 경로.
        rename_rule (function): 파일 이름 변경 규칙을 정의하는 함수.
        start_index (int): 변경할 파일의 시작 번호 (기본값: 1).
    """
    try:
        files = os.listdir(folder_path)

        for index, file_name in enumerate(files, start=start_index):
            old_path = os.path.join(folder_path, file_name)

            if os.path.isfile(old_path):
                new_name = rename_rule(file_name, index)
                new_path = os.path.join(folder_path, new_name)

                os.rename(old_path, new_path)
                print(f"Renamed: {file_name} -> {new_name}")

    except Exception as e:
        print(f"Error: {e}")


# 파일 이름 변경 규칙 (예: 'image_001.확장자' 형식으로 변경)
def rename_rule(file_name, index):
    extension = os.path.splitext(file_name)[1]
    number_str = str(index).zfill(3)
    return f"image_{number_str}{extension}"


start_index = 53
folder_path = "/data/ephemeral/home/aims_image_classifier/data/test"

rename_files_in_folder(folder_path, rename_rule, start_index)
