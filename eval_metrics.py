import torch
import os
from torch.utils.data import DataLoader
from training.RCDataset import RCDataset
from preprocessor.RCPreprocessor import RCPreprocessor
from training.model import PilotNet
from sklearn.metrics import classification_report

# ==========================================
# [설정 영역] 여기만 수정해서 두 번 실행하세요!
# ==========================================
# 1. 평가할 모델 파일 경로 (.pth)
# 예: "models/pilotnet_steering_20251128_211455.pth" (전처리 O 모델)
model_path = r"C:\autodrive\course-autodrive\models\pilotnet_steering_20251205_204611.pth"

# 2. 평가할 데이터셋의 CSV 파일
# 예: "datacollector/dataset_modified/data_labels_updated.csv" (전처리 O 데이터)
csv_filename = r"C:\autodrive\dataset\data_labels_updated.csv"
dataset_root = r"C:\autodrive\dataset"
# ==========================================

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 전처리 설정 (학습 때와 동일하게)
    preproc = RCPreprocessor(out_size=(200, 66), crop_top_ratio=0.4, crop_bottom_ratio=1.0)

    # 테스트 데이터셋 로드
    test_dataset = RCDataset(csv_filename=csv_filename, root=dataset_root, 
                             preprocessor=preproc, split="test", split_ratio=0.8)
    
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    # 모델 로드
    num_classes = len(test_dataset.angles)
    model = PilotNet(num_classes=num_classes, input_shape=(3, 66, 200)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds = []
    all_labels = []

    print("[INFO] 평가 진행 중...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 결과 출력 (Precision, Recall, F1-Score 포함)
    target_names = [str(angle) for angle in test_dataset.angles]
    print("\n" + "="*60)
    print(f" 모델: {os.path.basename(model_path)}")
    print(f" 데이터: {os.path.basename(csv_filename)}")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
    print("="*60)

if __name__ == "__main__":
    evaluate()