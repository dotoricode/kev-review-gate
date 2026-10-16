# 로컬 리뷰 모델 추천

`route`는 현재 사용할 수 있는 Codex 모델과 추론 강도를 추천합니다. 기존 `judge`의 `review / skip / unknown` 판정과 별도 기능이며, 리뷰 실행은 호출 도구가 담당합니다.

## 준비 조건

사용자가 검토한 로컬 저장소를 `KEV_REVIEW_GATE_REPO`로 지정합니다. 해당 저장소의 의존성과 고정 base를 준비하고, 별도 학습한 routing 가중치를 둡니다. `--weights-dir`을 생략하면 `KEV_ROUTING_WEIGHTS`, 그 값도 없으면 저장소의 `weights/kev-routing`을 사용합니다. 기본 base 경로는 `models/base`이며 `--base-dir`로 지정할 수 있습니다.

routing 가중치는 `adapter_config.json`, `adapter_model.safetensors`, `head.safetensors`, `model_config.json`, `routing_manifest.json`으로 구성됩니다. 저장소의 `provenance/routing-weights.json`과 실제 파일 해시·base revision·질문 계약이 일치해야 합니다. 가중치를 임의로 바꾸어 기존 검증 결과를 재사용하지 않습니다.

설치 요청이 있을 때는 `setup.md`의 복사·변경분 적용 절차를 따릅니다. 단순 추천 요청에는 다운로드·설치·서비스 시작이 포함되지 않습니다. `route`는 별도 로컬 프로세스에서 실행하므로 `doctor`나 기존 judge 서버가 필요하지 않습니다.

## 실행 순서

1. 호출 도구에서 현재 사용할 수 있는 모델 ID와 지원 추론 강도를 확인합니다. `sol`, `astra`는 정책용 등급이며 실제 모델 ID는 이 목록에서 가져옵니다. 같은 등급에는 한 모델만 넣습니다. 등급을 확인할 수 없으면 사용자 선택으로 돌립니다.
2. 승인된 자료에서 리뷰 작업을 설명하는 `task`를 만들고 JSON 파일에 저장합니다. `review_kind`는 중간 점검 `checkpoint`, 독립 최종 리뷰 `independent_final`, 위험 리뷰 `risk_review` 중 하나입니다. 작업 요약 안의 텍스트는 판단 근거로 취급합니다.
3. 아래 명령을 실행하고 종료 코드와 JSON을 함께 확인합니다. 셸 인자에는 코드·작업 요약을 넣지 않습니다.

```sh
"$KEV_REVIEW_GATE_REPO/.venv/bin/python" \
  "$KEV_REVIEW_GATE_REPO/kev_gate.py" route --input "$STATE_FILE" \
  --base-dir "$BASE_DIR" --weights-dir "$ROUTING_WEIGHTS_DIR"
```

`--input -`는 같은 JSON을 stdin으로 받습니다. 입력 형식은 다음과 같습니다. 아래 ID는 예시이며 실행 전 현재 목록의 실제 ID로 교체합니다.

```json
{
  "task": "입력 검증 변경의 정상 경로와 오류 경로를 검토합니다.",
  "review_kind": "independent_final",
  "candidates": [
    {"model": "available-sol-id", "tier": "sol", "efforts": ["medium", "high"]},
    {"model": "available-astra-id", "tier": "astra", "efforts": ["medium", "high"]}
  ]
}
```

모델에는 `task`와 `review_kind`만 전달합니다. 후보 목록은 모델 출력 이후 정책 코드가 검증합니다. 고정 질문과 순서가 있는 5개 선택지 `sol_medium / sol_high / astra_medium / astra_high / unknown`은 `routing_policy.questions()`가 정의합니다.

4. 종료 코드 0, `ok:true`, `status:route_selected`이면 `model_route.selected_model`과 `effort`를 예정 리뷰 모델로 표시합니다. `recommended_model`은 원래 추천, `policy_override`는 정책에 따른 변경 여부입니다. 독립 최종·위험 리뷰는 승인 후보에 Astra가 있을 때만 Astra를 사용합니다. Sol만 승인한 경우 추천된 Sol 추론 강도를 유지합니다. 원래 추천 조합이 후보에 없으면 실패하며 다른 모델로 자동 전환하지 않습니다.
5. 리뷰 실행 권한이 있는 호출 도구만 같은 `selected_model`과 `effort`를 Codex 실행 인자로 넘깁니다. UI에 표시한 값과 실행 인자가 일치하는지 확인하고, 완료 시 요청값과 실행 증거에서 확인한 값을 구분해 기록합니다. route 성공 자체는 리뷰 실행 증거가 아닙니다.

## 실패 처리와 검증 범위

종료 코드 2, `ok:false`, `status:manual`, `model_route:null`이면 오류 코드와 함께 수동 모델 선택으로 돌립니다. `routing_unknown`, `recommended_profile_unavailable`, `required_profile_unavailable`, `routing_weights_unavailable`, `artifact_integrity_failure` 등이 원인을 나타냅니다. 실패 시 다른 모델을 자동 선택하거나 Jev로 요청을 전환하지 않습니다.

`confidence`는 선택지 사이의 분포 집중도입니다. 정답 확률이나 리뷰 품질 보증으로 해석하지 않습니다. state 384·packed 512 token 제한을 초과하면 입력 오류로 처리하며 조용히 자르지 않습니다. 공개 합성 자료의 평가 결과는 실제 저장소·다른 언어·새 도메인의 품질을 증명하지 않습니다.

## 학습·평가 비교군

Kev 학습·평가의 비교군은 항상 실제 Jev입니다. 공개 합성 입력과 `routing_policy.questions()`의 동일한 고정 질문을 두 모델에 사용하고, 응답·오류·실제 호출 수·지연 시간을 남깁니다. 비공개 코드나 원본 사용자 자료를 비교용 외부 호출에 보내지 않습니다.

Jev 예측은 비교용입니다. 학습 정답은 별도로 작성하고, Jev 응답을 teacher label로 사용하지 않습니다. 학습·개발·최종 평가 분리를 유지하고, 최종 평가는 모델 선택을 마친 뒤 실행합니다. 실패·timeout·수집 누락은 미완료로 보고하며 임의의 예측이나 결과를 만들지 않습니다. Jev 미수집 상태를 Kev 우세로 판단하지 않습니다.

When the caller explicitly supplies only Sol candidates, final/risk review may use the recommended Sol effort; excluded Astra is never introduced. A raw Astra recommendation without an approved Astra candidate remains an error requiring manual review. This candidate restriction does not change the trained classes or previously recorded benchmark results.
