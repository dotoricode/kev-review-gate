# Kev 라우팅 학습과 Jev 비교 결과

Kev 라우팅용 LoRA·head를 실제로 학습하고, 고정 dev 평가로 **step25**를 선택했습니다. 학습·모델 선택에 사용하지 않은 final 20건에서 Kev는 **16/20(80%)**, 비교군 Jev는 **14/20(70%)**였습니다. 원래 고위험 4건의 과소 분류는 두 모델 모두 0건입니다. 기존 리뷰 필요성 판정용 `kev-review-gate-step50`은 공개 배포 해시와 일치하며 변경하지 않았습니다.

이 수치는 공개 합성 변경 설명에 대한 모델 프로파일 분류 결과입니다. 모델별 Codex 리뷰를 실행해 결함 발견률을 비교한 결과가 아닙니다. 20건에서 2건 차이이므로 일반적인 우월성을 입증하지 않습니다. 사전 최소 기준(정확도 0.8 이상·고위험 과소 분류 0건)은 충족하지만 실험 기능으로만 취급합니다.

| 평가 | Kev step0 | Kev step25 | Kev step50 | Jev |
| --- | ---: | ---: | ---: | ---: |
| dev 정확도 | 12/20, 60% | 18/20, 90% | 18/20, 90% | 16/20, 80% |
| dev 고위험 과소 분류 | 4/4 | 0/4 | 1/4 | 0/4 |
| final 정확도 | 평가 안 함 | 16/20, 80% | 평가 안 함 | 14/20, 70% |
| final 고위험 과소 분류 | 평가 안 함 | 0/4 | 평가 안 함 | 0/4 |
| final 응답 지연 중앙값 | 평가 안 함 | 111.625ms | 평가 안 함 | 232.4605ms |

지연은 모델 최초 로딩을 제외한 각 요청의 클라이언트 측 경과 시간입니다. Kev는 로컬 MLX, Jev는 TypeSafe 네트워크 API이므로 서로 다른 실행 환경의 측정값입니다. dev와 final 모두 누락 응답은 0건입니다. Jev는 `jev-latest`에 dev 20회·final 20회 실제 호출했으며, 응답의 전체 확률을 보존했습니다. Jev 응답은 학습 정답으로 사용하지 않았습니다. 별도 고정 리비전 이름이 API에 제공되지 않으므로 이후 `jev-latest` 결과가 같다고 보장하지 않습니다.

| final 분류 | Kev recall | Jev recall |
| --- | ---: | ---: |
| `sol_medium` | 4/4 | 4/4 |
| `sol_high` | 2/4 | 3/4 |
| `astra_medium` | 4/4 | 0/4 |
| `astra_high` | 4/4 | 4/4 |
| `unknown` | 2/4 | 3/4 |

Kev는 `sol_high`와 근거 부족을 보류하는 `unknown`에서 각각 2건을 놓쳤습니다. 최종·위험 리뷰의 Astra 강제 정책은 낮은 모델 등급 추천을 보정하지만, 적절한 추론 강도나 근거 부족 판단까지 보장하지 않습니다.

`raw_metrics`는 모델의 원래 5개 분류와 독립 합성 정답을 비교합니다. `policy_metrics`는 실제 리뷰 종류에 따른 Astra 최소 정책을 예측과 정답에 각각 적용한 뒤 비교합니다. final의 `policy_profile_underallocation`은 Kev 2건·Jev 1건이며, 정책이 올린 프로파일 목표까지 포함한 수치입니다. 이것을 원래 고위험 4건의 과소 분류로 해석하면 안 됩니다. `original_high_risk_downrouting`은 두 모델 모두 0/4입니다. dev 3개 보고서는 선택 전에 고정한 원본을 그대로 보존하므로 그 안의 policy `high_risk_cases`는 변환 후 `astra_high` 목표 수를 뜻합니다. 선택과 합격 판정에는 항상 raw 지표를 사용했습니다.

학습과 선택은 `data/protocol.json`을 사전에 고정해 수행했습니다. seed321, 학습80건(20개 의미 시나리오×표현4개), dev20건, final20건이며 의미 시나리오는 분할 간 겹치지 않습니다. 학습은 고정된 앞 50건으로 50 updates를 수행했습니다. 2-update smoke는 폐기하고 본학습을 원본으로 다시 초기화했습니다. 25→50에서는 optimizer와 CPU/MPS RNG를 이어받았습니다. step25와 step50의 dev 정확도가 같아 고위험 과소 분류가 적은 step25를 선택했습니다. final은 선택 파일의 해시를 고정한 후 각 모델 1회만 평가했고 점수에 따라 기준을 바꾸지 않았습니다.

원본은 `jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52`입니다. adapter·head·adapter config는 기존 lock SHA-256과 모두 일치했습니다. 본학습은 25 updates 61.419초, 다음 25 updates 57.510초가 걸렸습니다. 베이스 bfloat16·adapter/head float32, MPS, frozen base이며 두 구간 모두 adapter와 head가 실제로 변경됐습니다. 베이스 파라미터 SHA-256은 전후 `3e21c2093c9f34235845a6b3ccdce170f853112180d611c77b389cf050140ba9`로 같습니다. 전체 토큰 검사에서 최대 packed 길이는 train142·dev134·final136이며 512 제한을 자르지 않고 통과했습니다.

첫 smoke 시도는 학습 레코드의 필수 `src` 메타데이터 누락으로 실패했고 이를 보완했습니다. 다음 시도는 optimizer 실행 전 swap 증가가 512 MiB 한도를 넘어 안전 중단했습니다. 같은 중단 기준을 유지한 재시도에서 smoke와 본학습을 완료했습니다. 실패 기록도 최종 아티팩트에 보존합니다.

포장 과정에서 질문 해시의 JSON 직렬화를 기존 `contract.sha` 형식에 맞췄습니다. 이는 `model_config.json`의 메타데이터 정정이며, 고정 프로토콜·질문·학습 가중치는 바꾸지 않았습니다. 정정 이력은 `export-contract-amendment.json`에 있습니다. 선택 파일과 final 평가에서 사용한 manifest는 같습니다.

재현과 검토 근거는 다음 파일에 있습니다.

- `selected_checkpoint.json`: 선택 step·dev 보고서 해시·선택 가중치 manifest·선택 자체 해시
- `kev-dev-step*.json`, `jev-dev.json`, `kev-final.json`, `jev-final.json`: 입력 해시·확률·지연·raw 및 정책 지표
- `*.journal.jsonl`, `final-attempt-*.json`: 실제 호출 전 예약과 각 응답, final 1회 실행 기록
- `training-integrity.json`: 학습 무결성·원본 다운로드 해시·기존 review-gate 가중치 보존 확인
- `final-comparison.csv`: final 각 사례의 정답·두 모델 판단·정책 적용 후 모델과 추론 강도

잘못된 API 응답이 전체 평가를 중단하던 문제는 수정 전 테스트에서 재현했습니다. 응답 형태·확률·confidence·최대 확률 선택을 검사하고 오류 행은 `unavailable`로 보존합니다. final 재시도, 선택 파일 변조, 중복 case ID, 호출 중간 결과 유실을 막는 검사도 추가했습니다. 현재 관련 단위 검사 27개가 통과했습니다. 실제 운영 CLI reload와 Baby Steps 표시·실행 연결의 증거는 별도 통합 검증 결과에서 확인해야 합니다.

사전 전체 토큰 길이 검사는 final 입력 파일도 읽었습니다. final label을 optimizer 입력이나 checkpoint 선택에 사용하지 않았지만, 외부에서 봉인한 블라인드 평가였다고 주장하지 않습니다.

PR #1의 Codex 리뷰 지적 3건도 반영했습니다. 평가기는 텐서와 베이스를 로드하기 전에 실제 파일 SHA-256을 확인합니다. 운영 모델은 선택 step25에 고정하고, dev 평가에서만 저장소에 고정한 0·25·50 후보를 허용합니다. 선택 파일 생성 시 dev 데이터·질문·provider·후보 step·각 행의 입력 및 정답·확률·계산 지표를 확인합니다. `original_high_risk_downrouting`은 정책 적용 전 원래 예측에서 계산하며, 정책 적용 후 남는 위험 과소 분류는 `policy_residual_high_risk_downrouting`으로 구분합니다.

기존 측정 때에는 평가 로더의 사전 파일 해시 검증이 빠져 있었습니다. 이번 후속 검사에서 현재의 base와 세 후보의 실제 파일이 고정 manifest 및 기존 결과의 manifest와 일치함을 확인했습니다. dev 결과로 다시 계산한 선택은 기존 선택 파일과 같으며 선택·원본 결과의 해시는 보존했습니다. final이나 Jev 호출을 다시 실행하지 않았습니다. 정책 지표의 오프라인 재계산과 원본 파일 해시는 `pr1-review-correction-audit.json`에 별도로 보관했습니다. 최종 raw 정확도와 원래 위험 과소 분류 결과는 바뀌지 않았습니다.

PR #1의 두 번째 Codex 리뷰에서 확인된 학습 입력 검증과 `python -O` 문제도 수정했습니다. 이제 target0을 포함한 모든 실행은 Checkpoint·GPU 로드 전에 베이스 전체 파일과 원본 adapter·head·config를 기존 upstream lock 해시로 검사합니다. 프로토콜 파일 자체도 사전 고정 SHA로 검사하며, 데이터·질문·학습 대상·gradient·베이스 동결·adapter/head 변경의 필수 조건은 최적화 모드에서도 사라지지 않는 명시적 예외로 처리합니다. 새로운 25→50 실행은 이전 체크포인트와 optimizer/RNG 파일의 해시 및 완료 보고서의 provenance도 검증합니다. 이 해시 기록이 없는 과거 실행 폴더를 새 CLI로 재개하는 것은 거절합니다.

기존 본학습은 일반 Python으로 수행했고 원본 다운로드 해시는 별도로 확인했습니다. 본학습이나 final을 다시 실행하지 않았습니다. 현재 입력을 최적화 모드에서 다시 검사한 결과, 원본·베이스·고정 프로토콜이 모두 일치했습니다. 이 검사는 과거 로더에 없던 검사가 과거에도 실행됐다고 소급 주장하는 근거는 아닙니다. `pr1-round2-training-input-audit.json`에 후속 확인과 원본 결과 보존 해시를 기록했습니다.

PR #1의 세 번째 Codex 리뷰에서는 선택 파일의 자체 해시를 다시 계산하거나 선택 파일 경로를 복사하면 final 1회 조건을 우회할 수 있음을 수정 전 테스트로 재현했습니다. 이제 final은 코드에 별도로 고정한 선택 SHA와 일치하는 파일만 허용합니다. 새로운 final 잠금은 임의 선택 파일의 위치가 아니라 저장소의 고정 실험 위치에서 protocol·selection 해시·provider로 구분합니다. 기존 `final-attempt-kev.json`, `final-attempt-jev.json`도 사용 완료로 처리하며 덮어쓰거나 삭제하지 않습니다. 다른 실험을 진행하려면 별도의 검토된 고정값과 실험 기록이 필요합니다.

실제 저장된 선택 파일을 다른 경로로 복사한 경우에도 두 provider 모두 기존 사용 완료 잠금으로 거절됨을 오프라인에서 확인했습니다. dev로 재도출한 선택이 별도 고정 SHA와 같고, 선택·final 결과·기존 잠금 파일의 전후 해시도 같았습니다. 학습·final·Jev 호출은 추가로 실행하지 않았습니다. 확인 내용은 `pr1-round3-final-guard-audit.json`에 있습니다. 전체 회귀 검사는 93개가 통과했습니다.

PR #1의 네 번째 Codex 리뷰에서는 protocol의 데이터 해시 역할을 바꿔 final을 dev로 위장하는 우회를 수정 전 재현했습니다. 평가 CLI·dev 보고서 검증·선택 파일 생성은 모두 공통 로더에서 protocol 원본 바이트의 고정 SHA를 먼저 확인합니다. 따라서 protocol 안의 해시 역할이나 다른 설정을 수정하면 provider 호출 전에 거절합니다. 데이터 파일은 한 번 읽은 바이트를 해시·역할 분류·사례 파싱·결과 기록에 함께 사용합니다. 해시 확인 뒤 파일을 다시 읽는 틈에 final 내용으로 바뀌는 문제도 수정 전 재현해 막았습니다. 이름을 바꾼 final도 원래 데이터 해시로 식별돼 사용 완료 잠금을 거칩니다.

수정 후 protocol와 dev 선택을 오프라인으로 다시 검증했고, 원본 선택·final·사용 완료 잠금 파일의 해시는 유지됐습니다. 모델·학습·Jev를 추가 호출하지 않았습니다. 관련 근거는 `pr1-round4-protocol-guard-audit.json`에 있으며 전체 회귀 검사 97개가 통과했습니다.

PR #1의 다섯 번째 Codex 리뷰에서는 학습 데이터 검증 후 파일을 다시 읽으면 다른 내용을 토큰 검사나 최적화에 사용할 수 있음을 지적했습니다. protocol도 해시 확인 뒤 별도로 읽으면 변경된 설정을 파싱할 수 있어 수정 전 테스트로 재현했습니다. 이제 protocol과 각 데이터셋은 한 번 읽은 바이트를 해시 확인과 JSON 파싱에 함께 사용합니다. 검증한 사례 목록을 메모리에 보관해 토큰 사전 검사와 optimizer 입력에서 같은 목록을 사용하며, 학습 보고서의 protocol 해시도 검증한 값을 기록합니다. 기존 `verify_protocol` 호출의 기본 반환 형식은 유지합니다.

학습·평가 모듈의 `read_*` 경로를 함께 점검했습니다. 평가의 dev 검증은 이미 검증한 `dev_bytes`를 직접 파싱하고 있었으며, 파일을 다시 읽을 때 다른 내용이 반환되는 회귀 조건에서도 통과함을 별도 검사로 확인했습니다. 원본 데이터·protocol·dev 보고서·선택·final·사용 완료 잠금의 해시가 유지됐고 모델·학습·Jev를 추가 호출하지 않았습니다. 근거는 `pr1-round5-training-snapshot-audit.json`이며 전체 회귀 검사 100개가 통과했습니다.
