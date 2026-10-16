# Kev 공동 판정 v2 실험

리뷰 필요성(`review/skip/unknown`)과 Sol 추론 강도(`sol_medium/sol_high/unknown`)를 같은 구조화 근거에서 함께 학습합니다. Astra는 선택지에도 후보에도 포함하지 않습니다. `skip`은 모델의 참고 의견이며 자동 리뷰 생략 승인이나 필수 리뷰 정책 변경이 아닙니다.

기존 need 모델과 router-v1 가중치·소비된 final은 보존합니다. v2는 별도 adapter/head이며 `discriminate` 명령으로만 선택합니다. 기본 활성화나 v1 실패 시 자동 전환은 하지 않습니다.

공개 합성 train 48건(12 family), dev 16건(4 family), final 16건(4 family)을 의미 시나리오 단위로 분리합니다. 각 family는 비실행 변경, 단순 동작 변경, 민감 동작 변경, 정보 충돌 네 사례를 포함합니다. 같은 family에서 `changes` 외 보조 필드는 동일합니다. 테스트 통과 여부만으로 민감 변경을 낮게 평가하거나 컴포넌트 이름만으로 문구 변경을 높게 평가하지 않는지 확인합니다. 영어와 한국어가 섞인 입력을 포함하며 완전한 한국어 검증은 아닙니다.

가중치와 질문은 공개 원본과 고정 revision을 사용합니다. frozen base bfloat16, adapter/head float32, AdamW lr2e-5, warmup10, clipping1, batch1, seed722는 기존 학습기와 같습니다. 50 updates에서 모든 48개 train 행을 한 번 이상 사용합니다. 25→50은 같은 optimizer와 CPU/MPS RNG를 이어 갑니다. 원본 초기화 step0, step25, step50을 모두 dev에서 평가합니다. base SHA-256 전후 일치와 adapter/head gradient·실제 변경을 기록합니다.

선택 점수는 `(need macro recall + route macro recall) / 2` 최대입니다. 동률이면 민감 사례에서 need가 review가 아니거나 route가 high가 아닌 수를 최소화하고, 다시 동률이면 이른 step을 선택합니다. 미수집 사례는 recall 분모에 남습니다. 네 사례가 모두 두 문항 정답인 family 비율을 paired correct fraction으로 계산합니다. 누락 정보에서 어느 문항이든 unknown이 아니면 false precision입니다.

`data/protocol.json`에 고정한 승격 기준은 need/route macro recall 각각 0.8 이상, 민감 과소 판단 0, 누락 정보 false precision 0, 비실행 변경의 잘못된 review 0, family 네 쌍 정답 비율 0.75 이상, 모든 사례 수집입니다. 기준 미달이면 v2 운영 pin을 만들지 않습니다. 임계값으로 답을 강제로 나누지 않습니다.

비교군은 실제 `jev-latest` API입니다. 같은 state와 두 questions를 dev 16·final 16건에서 각각 한 번 요청하며, Jev 응답은 학습 정답이 아닙니다. dev 선택 뒤 final은 provider마다 한 번만 실행하고 오류도 재요청하지 않습니다. always-review/high와 always-review/medium 상수 기준도 계산합니다. v1의 다른 질문 계약과의 엄밀한 동일 입력 성능 비교는 이 실험에 포함하지 않습니다. step0은 새 질문에서 원본 가중치를 평가한 학습 전 기준입니다.

사례·정답은 사람이 정의한 합성 정책 기대입니다. 균형 잡힌 소표본과 반복 구조는 운영 분포, 실제 결함 발견률이나 운영 정확도를 증명하지 않습니다. final 토큰 길이는 학습 전에 확인하지만 final 결과는 체크포인트 선택이나 학습에 사용하지 않습니다.

가용 메모리 4 GiB 미만, 실행 시작 대비 swap 512 MiB 초과 증가, 15분 초과 시 중단합니다. state384·packed512 제한을 넘으면 거절하며 입력을 임의로 자르지 않습니다. 중단 시 증거와 기존 가중치를 유지합니다.
