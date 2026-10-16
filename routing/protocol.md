# Kev 라우팅 학습·Jev 비교 프로토콜

이 실험은 공개 합성 변경 설명을 보고 리뷰 모델 등급과 추론 강도를 분류하는 능력을 측정합니다. 실제 Codex 리뷰를 모델별로 실행해 결함 발견률을 비교하는 실험은 아닙니다. 비교군은 항상 실제 TypeSafe API의 `jev-latest`입니다. Jev 결과를 학습 정답으로 사용하지 않습니다.

`data/protocol.json`은 학습과 API 호출 전에 고정했습니다. seed는 `321`입니다. 학습 80건은 20개 의미 시나리오의 표현 변형입니다. dev 20건과 final 20건은 학습과 다른 의미 시나리오입니다. 각 분할에서 5개 분류의 개수는 같습니다. task에는 정답 식별자를 넣지 않습니다.

| 구분 | 고정 조건 |
| --- | --- |
| 판단 계약 | `routing_policy.questions()`의 단일 `route` 질문과 5개 선택지 |
| 모델 입력 | `task`, `review_kind`; 모델 후보 목록은 결정적 정책이 별도로 처리 |
| 학습 원본 | `jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52` |
| 베이스 | `Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b` |
| 갱신 대상 | LoRA rank16와 pointer head; 베이스 동결 |
| 정밀도 | 베이스 bfloat16, adapter·head float32; MPS, CPU fallback 금지 |
| 학습량 | batch 1, 최대 50 updates; 80개 학습 행 중 앞의 고정된 50개 사용 |
| 선택 | step 0·25·50의 dev macro accuracy 최대, 동률이면 위험 과소 분류 최소, 이후 빠른 step |
| final | dev 선택 후 선택 모델과 Jev를 같은 final 20건에서 각 1회 평가 |
| 제한 | state 384·packed 512 tokens, 초과 입력 거절, 자르기 금지 |
| 호출 예산 | Jev dev 20 + final 20; 오류를 정답으로 대체하지 않고 재호출하지 않음 |
| 안전 중단 | available memory 4 GiB 미만, 실행 시작 대비 swap 512 MiB 증가 초과, segment 15분 초과 |

본학습 전 2 update smoke의 가중치는 버립니다. 본학습은 원본으로 다시 초기화합니다. 25→50 구간에서는 같은 실행의 optimizer와 CPU/MPS RNG를 이어받습니다. 베이스 파라미터의 SHA-256 전후 일치, adapter·head gradient와 실제 변경 여부를 기록합니다.

Jev와 Kev에 동일한 state와 question을 전달하며 입력 해시, 선택, 전체 확률, 지연을 보관합니다. API 실패·확률 누락은 `unavailable`이며 정확도 측정 건수와 분리합니다. 위험 과소 분류는 정답 `astra_high`를 `sol_medium`, `sol_high`, `astra_medium`으로 판단한 경우입니다. `unknown`은 별도 보류이며 위험 과소 분류에서 제외합니다.

final 정확도 0.8 미만 또는 위험 과소 분류가 하나라도 있으면 기본 라우팅 활성화를 막습니다. 통과하더라도 소규모 합성 실험이므로 실험 기능으로만 취급합니다. 기존 리뷰 필요성 판단용 `step50`의 가중치와 평가는 바꾸지 않습니다.
