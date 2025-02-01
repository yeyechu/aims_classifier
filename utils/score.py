import pandas as pd

# CSV 파일 경로
file_path = "/data/ephemeral/home/aims_image_classifier/data/inference_labels_teacher.csv"

# CSV 파일 읽기
df = pd.read_csv(file_path, header=None, names=["filename", "true_label", "predicted_label", "confidence"])

# 정답 레이블과 예측 레이블이 일치하는 개수 계산
correct_count = (df["true_label"] == df["predicted_label"]).sum()

# 전체 데이터 개수
total_count = len(df)

# 정확도 계산
accuracy = correct_count / total_count

# 정확도를 파일의 마지막 행에 추가
with open(file_path, "a") as f:
    f.write(f"\nAccuracy: {accuracy:.4f}")

print(f"정확도 계산 완료: {accuracy:.4f} (파일에 저장됨)")
