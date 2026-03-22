import os

os.chdir(os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.optim as optim
import math

# =========================
# 1. 커스텀 16차원 임베딩
# =========================
phoneme_to_vec = {
    # --- 단자음 ---
    "ㄱ": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "ㄴ": [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "ㄷ": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "ㄹ": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "ㅁ": [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "ㅂ": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "ㅅ": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "ㅇ": [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "ㅈ": [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "ㅊ": [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    "ㅋ": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    "ㅌ": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    "ㅍ": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    "ㅎ": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # --- 된소리 ---
    "ㄲ": [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    "ㄸ": [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    "ㅃ": [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    "ㅆ": [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "ㅉ": [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    # --- 겹받침 ---
    "ㄳ": [1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    "ㄵ": [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    "ㄶ": [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "ㄺ": [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    "ㄻ": [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "ㄼ": [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    "ㄽ": [1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "ㄾ": [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
    "ㄿ": [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
    "ㅀ": [1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "ㅄ": [1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    # --- 단모음 ---
    "ㅏ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "ㅓ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "ㅗ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    "ㅜ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "ㅡ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "ㅣ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "ㅐ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "ㅔ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "ㅚ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    "ㅟ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    # --- 복합모음 ---
    "ㅘ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    "ㅙ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
    "ㅛ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    "ㅝ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    "ㅞ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
    "ㅠ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    "ㅢ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    "ㅑ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    "ㅕ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "ㅒ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    "ㅖ": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    # --- 특수 ---
    "<pad>": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "<end>": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}

EMBED_DIM = 16
MAX_SEQ = 10  # 최대 몇 개 토큰까지 볼지

tokens = list(phoneme_to_vec.keys())
token_to_id = {t: i for i, t in enumerate(tokens)}
id_to_token = {i: t for t, i in token_to_id.items()}
vocab_size = len(tokens)


def seq_to_tensor(token_list: list) -> torch.Tensor:
    """
    토큰 리스트 → (seq_len, 1, 16) 텐서
    ex) ["ㅎ", "ㅏ", "ㄴ"] → shape (3, 1, 16)
    """
    vecs = [phoneme_to_vec[t] for t in token_list]
    return torch.tensor(vecs, dtype=torch.float32).unsqueeze(1)  # (seq, 1, dim)


# =========================
# 2. Attention 모듈
# =========================
class SingleHeadAttention(nn.Module):
    """
    경량 Single-Head Attention
    파라미터: W_Q(16x16) + W_K(16x16) + W_V(16x16) = 768개
    """

    def __init__(self, dim=EMBED_DIM):
        super().__init__()
        self.W_Q = nn.Linear(dim, dim, bias=False)  # 질문 변환
        self.W_K = nn.Linear(dim, dim, bias=False)  # 답변 변환
        self.W_V = nn.Linear(dim, dim, bias=False)  # 내용 변환
        self.scale = math.sqrt(dim)  # 유사도 스케일 조정

    def forward(self, x):
        # x: (seq_len, 1, 16)
        Q = self.W_Q(x)  # (seq_len, 1, 16) — 질문
        K = self.W_K(x)  # (seq_len, 1, 16) — 답변
        V = self.W_V(x)  # (seq_len, 1, 16) — 내용

        # Q·K^T: 각 토큰끼리 유사도 계산
        # (seq_len, 16) x (16, seq_len) = (seq_len, seq_len)
        Q_ = Q.squeeze(1)  # (seq_len, 16)
        K_ = K.squeeze(1)  # (seq_len, 16)
        V_ = V.squeeze(1)  # (seq_len, 16)

        scores = torch.matmul(Q_, K_.T) / self.scale  # (seq_len, seq_len)

        # 미래 토큰은 보면 안 됨 → 마스킹 (현재 이전만 볼 수 있게)
        seq_len = x.size(0)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)  # (seq_len, seq_len)

        # 가중치 × V → 문맥이 담긴 벡터
        out = torch.matmul(weights, V_)  # (seq_len, 16)
        return out.unsqueeze(1)  # (seq_len, 1, 16)


# =========================
# 3. 모델
# =========================
class LightAttentionLM(nn.Module):
    """
    전체 파라미터:
      Attention : 16×16 × 3 = 768개
      fc1       : 16×32     = 512개
      fc2       : 32×vocab  = 32×58 ≈ 1856개
      합계       : ~3136개   ← 극도로 가벼움
    """

    def __init__(self, vocab_size, dim=EMBED_DIM, hidden=32):
        super().__init__()
        self.attention = SingleHeadAttention(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        # x: (seq_len, 1, 16)
        ctx = self.attention(x)  # (seq_len, 1, 16) — 문맥 반영된 벡터
        last = ctx[-1]  # (1, 16) — 마지막 토큰의 문맥 벡터만 사용
        h = self.relu(self.fc1(last))  # (1, hidden)
        out = self.fc2(h)  # (1, vocab_size)
        return out


model = LightAttentionLM(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# 파라미터 수 출력
total_params = sum(p.numel() for p in model.parameters())
print(f"총 파라미터 수: {total_params}개")

# =========================
# 4. 학습 데이터
# =========================
sentences = [
    ["ㅎ", "ㅏ", "ㄴ", "ㅡ", "ㄹ", "<end>"],  # 하늘
    ["ㅂ", "ㅏ", "ㄹ", "ㅏ", "ㅁ", "<end>"],  # 바람
    ["ㄱ", "ㅏ", "ㄹ", "ㅡ", "ㄹ", "<end>"],  # 가을
    ["ㄴ", "ㅏ", "ㅁ", "ㅜ", "<end>"],  # 나무
    ["ㅅ", "ㅏ", "ㄹ", "ㅏ", "ㅇ", "<end>"],  # 사랑
]


def make_training_data(sentences):
    """
    문장 → (입력 시퀀스, 정답) 쌍 생성
    ex) ["ㅎ","ㅏ","ㄴ","ㅡ","ㄹ","<end>"]
        → (["ㅎ"],           "ㅏ")
           (["ㅎ","ㅏ"],      "ㄴ")
           (["ㅎ","ㅏ","ㄴ"], "ㅡ")  ...
    """
    data = []
    for sent in sentences:
        for i in range(len(sent) - 1):
            input_seq = sent[: i + 1]  # 처음부터 현재까지 전부
            target = sent[i + 1]  # 다음 토큰
            data.append((input_seq, target))
    return data


training_data = make_training_data(sentences)

# =========================
# 5. 학습
# =========================
for epoch in range(500):
    total_loss = 0
    for input_seq, target in training_data:
        x = seq_to_tensor(input_seq)  # (seq_len, 1, 16)
        y = torch.tensor([token_to_id[target]])
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 100 == 0:
        print(f"epoch {epoch}, loss {total_loss:.4f}")


# =========================
# 6. 생성 함수
# =========================
def generate(start_token, max_len=10):
    result = [start_token]
    for _ in range(max_len):
        x = seq_to_tensor(result)  # 지금까지 생성된 전체 시퀀스
        pred = model(x)
        next_id = torch.argmax(pred).item()
        next_token = id_to_token[next_id]
        if next_token == "<end>":
            break
        result.append(next_token)
    return " → ".join(result)


# =========================
# 7. 테스트
# =========================
print()
print("ㅎ 으로 시작:", generate("ㅎ"))  # 하늘
print("ㅂ 으로 시작:", generate("ㅂ"))  # 바람
print("ㄴ 으로 시작:", generate("ㄴ"))  # 나무
