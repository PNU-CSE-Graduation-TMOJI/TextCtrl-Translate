import cv2
import matplotlib.pyplot as plt

# 원본 이미지 읽기
path = "./example_result/00000.png"
img = cv2.imread(path)

# --- Step 1: 노이즈 제거 ---
denoised = cv2.fastNlMeansDenoisingColored(img, None, h=15, hColor=15, templateWindowSize=7, searchWindowSize=21)

# --- Step 2: 샤프닝 (Unsharp Masking) ---
gaussian = cv2.GaussianBlur(denoised, (0,0), 3)
sharp = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)

# --- Step 3: 밝기 / 대비 조정 ---
# alpha = 대비(1.0~2.0 정도 추천), beta = 밝기(0~50 정도 추천)
bright = cv2.convertScaleAbs(sharp, alpha=1.05, beta=20)

# --- 시각화 ---
plt.figure(figsize=(20,5))
plt.subplot(1,4,1); plt.title("Original"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.subplot(1,4,2); plt.title("Denoised"); plt.imshow(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.subplot(1,4,3); plt.title("Denoised+Sharpened"); plt.imshow(cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.subplot(1,4,4); plt.title("Brightened"); plt.imshow(cv2.cvtColor(bright, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.show()

# --- 저장 ---
cv2.imwrite("./imgs/00000_denoised.png", denoised)
cv2.imwrite("./imgs/00000_denoised_sharp.png", sharp)
cv2.imwrite("./imgs/00000_final.png", bright)

print("✔ 노이즈 제거 결과: 00000_denoised.png")
print("✔ 선명하게 보정한 결과: 00000_denoised_sharp.png")
print("✔ 밝기/대비 보정 최종 결과: 00000_final.png")

# 색이 뭔가 샤프하지 못하다.. 이 부분 후처리하면 좋을듯