// =====================================================================
// analyze.js — 페이지 새로고침 없이 분석 요청하고, 결과 한 줄을 표 맨 위에 추가
//
// 서버는 동기 방식 그대로 (Celery 없음):
//   fetch 요청 → Django가 FastAPI 결과까지 받고 → JSON 응답 → 이 파일이 표에 추가
//
// 처음 표는 views.index(GET)가 템플릿으로 그리고, 새 결과만 이 파일이 추가함
// =====================================================================

const form = document.getElementById("analysis-form");
const button = form.querySelector("button");
const errorBox = document.getElementById("error-box");
const tbody = document.getElementById("job-rows");

// ★ 백엔드 JSON 주소 : 템플릿의 data-api 값 (예: /analyze/)
//   [연결] project10/urls.py → path("analyze/", views.analyze, name="analyze")
const API_URL = form.dataset.api;

form.addEventListener("submit", async (e) => {
  e.preventDefault();                                   // 폼 기본 동작(페이지 이동) 막기

  // ★ name="video" : views.py의 request.FILES.get("video") 와 같아야 함
  const fileInput = form.querySelector("[name=video]");
  if (fileInput.files.length === 0) {
    showError("영상을 선택하세요.");
    return;
  }

  button.disabled = true;
  button.textContent = "분석 중...";
  showError("");

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      // ★ csrfmiddlewaretoken : 템플릿의 {% csrf_token %} 이 만든 값 (Django CSRF 검사용)
      headers: { "X-CSRFToken": form.querySelector("[name=csrfmiddlewaretoken]").value },
      body: new FormData(form),                         // 영상 파일 포함
    });
    const job = await res.json();

    if (!res.ok) {
      // ★ error : views.py가 실패 시 돌려주는 JSON 키 (예: {"error": "영상을 선택하세요."})
      showError(job.error || `요청 실패 (${res.status})`);
      return;
    }

    addRow(job);
    form.reset();
  } catch (err) {
    showError("서버에 연결할 수 없습니다.");
  } finally {
    button.disabled = false;
    button.textContent = "AI 분석 시작";
  }
});

// ---------------------------------------------------------------------
// 표 맨 위에 한 줄 추가
// ★ JSON 키 6개 : views.py의 JsonResponse({...}) 키와 같아야 함
//   id · action_label · confidence · file_type · status · processing_time
// ---------------------------------------------------------------------
function addRow(job) {
  const empty = document.getElementById("empty-row");   // "아직 작업 기록이 없어요" 줄
  if (empty) empty.remove();

  const tr = document.createElement("tr");
  tr.appendChild(cell(job.id));
  tr.appendChild(cell(job.action_label || "-"));
  tr.appendChild(cell(job.confidence != null ? Number(job.confidence).toFixed(2) : "-"));
  tr.appendChild(cell(job.file_type || "-"));

  const statusTd = document.createElement("td");
  const badge = document.createElement("span");
  badge.className = `badge badge-${job.status}`;        // 템플릿 CSS의 배지 색 그대로 사용
  badge.textContent = job.status;
  statusTd.appendChild(badge);
  tr.appendChild(statusTd);

  tr.appendChild(cell(job.processing_time != null ? `${Number(job.processing_time).toFixed(1)}초` : "-"));

  tbody.prepend(tr);                                    // 최신 작업이 맨 위
}

function cell(text) {
  const td = document.createElement("td");
  td.textContent = text;                                // 글자로만 넣기 (HTML로 해석되지 않게)
  return td;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = !message;
}