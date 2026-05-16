const form = document.getElementById('predict-form')
const resultsEl = document.getElementById('results')
const priceSummary = document.getElementById('summaryPrice')
const confidenceSummary = document.getElementById('summaryConfidence')
const ctx = document.getElementById('predChart')
const themeToggleBtn = document.getElementById('themeToggle')
const htmlEl = document.documentElement

let chart = null
const API_BASE_URL = ''

// Initialize Theme
const storedTheme = localStorage.getItem('theme')
const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
if (storedTheme === 'dark' || (!storedTheme && prefersDark)) {
  htmlEl.classList.add('dark')
} else {
  htmlEl.classList.remove('dark')
}

// Initialize History
let predictionHistory = JSON.parse(localStorage.getItem('goldPredictionHistory')) || []

themeToggleBtn.addEventListener('click', () => {
  htmlEl.classList.toggle('dark')
  const isDark = htmlEl.classList.contains('dark')
  localStorage.setItem('theme', isDark ? 'dark' : 'light')
  renderChart(predictionHistory) // Re-render chart to update grid/text colors
})

form.addEventListener('submit', async (e) => {
  e.preventDefault()
  const payload = {
    weight_grams: parseFloat(document.getElementById('weight_grams').value || 50.0),
    purity_karat: document.getElementById('purity_karat').value,
    color: document.getElementById('color').value,
    finish: document.getElementById('finish').value,
    certification: document.getElementById('certification').value,
    hallmark: document.getElementById('hallmark').value,
    x: parseFloat(document.getElementById('x').value || 20.0),
    y: parseFloat(document.getElementById('y').value || 18.5),
    z: parseFloat(document.getElementById('z').value || 6.5),
    depth: parseFloat(document.getElementById('depth').value || 28.0),
    table: parseFloat(document.getElementById('table').value || 55.0)
  }

  resultsEl.innerHTML = `<div class='rounded-3xl bg-slate-100 dark:bg-slate-950/90 px-5 py-4 text-slate-800 dark:text-slate-300 shadow-inner dark:shadow-slate-950/50'>Predicting price...</div>`

  let data
  try {
    const resp = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
    })

    if (!resp.ok) {
      const errorText = await resp.text()
      throw new Error(`Server returned ${resp.status}: ${errorText}`)
    }

    data = await resp.json()
  } catch (error) {
    resultsEl.innerHTML = `<div class='rounded-3xl border border-rose-500 bg-rose-50 dark:bg-rose-950/80 px-5 py-4 text-rose-800 dark:text-rose-300 shadow-inner dark:shadow-rose-900/20'>Network/API error: ${error.message}</div>`
    priceSummary.textContent = '--'
    confidenceSummary.textContent = 'Prediction failed. Check the backend server or connection.'
    return
  }

  if (data.error) {
    resultsEl.innerHTML = `<div class='rounded-3xl border border-rose-500 bg-rose-50 dark:bg-rose-950/80 px-5 py-4 text-rose-800 dark:text-rose-300 shadow-inner dark:shadow-rose-900/20'>Error: ${data.error}</div>`
    priceSummary.textContent = '--'
    confidenceSummary.textContent = 'Prediction failed. Check the inputs.'
    return
  }

  const r = data.results[0]
  const predValue = Number(r.prediction)
  
  // Update History
  predictionHistory.push(predValue)
  if (predictionHistory.length > 20) {
    predictionHistory.shift() // Keep only last 20 predictions
  }
  localStorage.setItem('goldPredictionHistory', JSON.stringify(predictionHistory))

  const formatted = `$${predValue.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2})}`
  const confidenceText = r.confidence !== null ? `${Number(r.confidence).toFixed(2)} (lower is more precise)` : 'N/A'

  resultsEl.innerHTML = `
    <div class='rounded-3xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950/90 p-5 shadow-xl shadow-slate-200 dark:shadow-slate-950/40'>
      <div class='flex items-center justify-between gap-4'>
        <div>
          <p class='text-sm uppercase tracking-[0.24em] text-slate-500'>Predicted gold price</p>
          <p class='mt-3 text-4xl font-semibold text-slate-900 dark:text-white'>${formatted}</p>
        </div>
        <div class='rounded-3xl bg-slate-50 dark:bg-slate-800/90 px-4 py-3 text-right'>
          <p class='text-sm text-slate-500 dark:text-slate-400'>Confidence</p>
          <p class='mt-1 text-xl font-semibold text-cyan-600 dark:text-cyan-300'>${confidenceText}</p>
        </div>
      </div>
    </div>
  `

  priceSummary.textContent = formatted
  confidenceSummary.textContent = `Confidence score: ${confidenceText}`
  renderChart(predictionHistory)
})

function renderChart(values) {
  if (!ctx) return
  
  const isDark = htmlEl.classList.contains('dark')
  const textColor = isDark ? '#cbd5e1' : '#475569'
  const gridColor = isDark ? 'rgba(148,163,184,0.2)' : 'rgba(148,163,184,0.3)'
  const pointColor = isDark ? 'rgba(125, 211, 252, 1)' : 'rgba(2, 132, 199, 1)'
  const lineColor = isDark ? 'rgba(96, 165, 250, 0.9)' : 'rgba(56, 189, 248, 0.9)'
  const areaColor = isDark ? 'rgba(56, 189, 248, 0.1)' : 'rgba(56, 189, 248, 0.2)'

  // If history is empty, show a single 0 to keep the chart rendered
  const chartData = values.length > 0 ? values : [0]
  const labels = chartData.map((_, i) => `Request ${i + 1}`)

  const data = {
    labels: labels,
    datasets: [{
      label: 'USD',
      data: chartData,
      fill: true,
      backgroundColor: areaColor,
      borderColor: lineColor,
      pointBackgroundColor: pointColor,
      borderWidth: 2,
      tension: 0.3 // Adds slight curve to the line
    }]
  }

  const options = {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: { enabled: true, callbacks: { label: context => `${context.formattedValue} USD` } }
    },
    scales: {
      x: { grid: { display: false }, ticks: { color: textColor } },
      y: { grid: { color: gridColor }, ticks: { color: textColor } }
    }
  }

  if (chart) {
    chart.data = data
    chart.options = options
    chart.update()
    return
  }

  chart = new Chart(ctx, { type: 'line', data, options })
}

window.addEventListener('load', () => {
  renderChart(predictionHistory)
})
