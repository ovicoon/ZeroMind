import os

os.chdir(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# 1. 토큰 정의
# =========================
tokens = ["0", "1", "<end>"]
token_to_id = {t: i for i, t in enumerate(tokens)}
id_to_token = {i: t for t, i in token_to_id.items()}
vocab_size = len(tokens)  # 3

# =========================
# 2. 고정 2D 커스텀 임베딩 벡터
# =========================
token_to_vec = {
    "0": [1.0, 0.0, 0.0],  # 이진 벡터: 0 → [1, 0, 0]
    "1": [0.0, 1.0, 0.0],  # 이진 벡터: 1 → [0, 1, 0]
    "<end>": [0.0, 0.0, 1.0],  # 종료 토큰
}


def token_to_tensor(token: str) -> torch.Tensor:
    """토큰 → shape (1, 3) float 텐서 (고정값, 학습 안 됨)"""
    return torch.tensor([token_to_vec[token]], dtype=torch.float32)


# =========================
# 3. 학습 데이터 (이진 시퀀스 패턴)
# =========================
sentences = [
    ["0", "1", "0", "1", "<end>"],  # 0→1 교대 패턴
    ["1", "0", "1", "0", "<end>"],  # 1→0 교대 패턴
    ["0", "0", "1", "1", "<end>"],  # 00→11 패턴
    ["1", "1", "0", "0", "<end>"],  # 11→00 패턴
]


# =========================
# 4. 모델 (입력: 고정 2D 벡터 → 출력: vocab 분류)
# =========================
class SimpleLM(nn.Module):
    def __init__(self, vocab_size, input_dim=3, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        h = self.relu(self.fc1(x))  # (1, hidden_dim)
        out = self.fc2(h)  # (1, vocab_size)
        return out


model = SimpleLM(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# =========================
# 5. 학습
# =========================
for epoch in range(300):
    total_loss = 0
    for sent in sentences:
        for i in range(len(sent) - 1):
            x = token_to_tensor(sent[i])  # (1, 2) 고정 벡터
            y = torch.tensor([token_to_id[sent[i + 1]]])  # 정답 토큰 id
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    if epoch % 50 == 0:
        print(f"epoch {epoch}, loss {total_loss:.4f}")


# =========================
# 6. 생성 함수
# =========================
def generate(start_token, max_len=10):
    current = start_token
    result = [current]
    for _ in range(max_len):
        x = token_to_tensor(current)
        pred = model(x)
        next_id = torch.argmax(pred).item()
        next_token = id_to_token[next_id]
        if next_token == "<end>":
            break
        result.append(next_token)
        current = next_token
    return " → ".join(result)


# =========================
# 7. 테스트
# =========================
print("0 으로 시작:", generate("0"))
print("1 로 시작:", generate("1"))
