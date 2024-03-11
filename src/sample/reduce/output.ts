import Chart from 'chart.js/auto';
import test from 'node:test';

type TableRowRecord = {
  algorithm: string;
  numPoints: number;
  workgroupSize: number;
  numDispatches: number;
  passed: boolean;
};

export type TestCase = {
  algorithm: string;
  numPoints: number;
  workgroupSize: number;
  numDispatches: number;
  data: Float32Array;
  execsPerSecond: number;
};

/**
 * Adds a div element to the output area (that normally would contain a canvas).
 * This element is used as the container for the output table and chart.
 * @param parentNode Parent node that normally contains the output canvas
 */
export function appendOutputElements(parentNode: Node): void {
  appendStylesheet();

  const topLevelElement = ensureTopLevelElement(parentNode);

  appendOutputTableElement(topLevelElement);
}

/**
 * Adds a canvas element with the benchmark results chart to the output area.
 */
export function appendOutputBenchmarksCharts(
  testCases: TestCase[],
  plotConfig: string
): void {
  function removeDuplicates(arr: string[]) {
    const unique = [];
    arr.forEach((element) => {
      if (!unique.includes(element)) {
        unique.push(element);
      }
    });
    return unique;
  }

  const plotParameters = plotConfig.split('-');

  const perPlotParameter = plotParameters[0];
  const perSeriesParameter = plotParameters[1];
  const independentParameter = plotParameters[2];

  let perPlotOptions = [];
  let perSeriesOptions = [];
  let independentOptions = [];

  for (const t of testCases) {
    perPlotOptions.push(t[perPlotParameter]);
    perSeriesOptions.push(t[perSeriesParameter]);
    independentOptions.push(t[independentParameter]);
  }

  perPlotOptions = removeDuplicates(perPlotOptions);
  perSeriesOptions = removeDuplicates(perSeriesOptions);
  independentOptions = removeDuplicates(independentOptions);

  for (const ppo of perPlotOptions) {
    let chartElement = document.querySelector<HTMLCanvasElement>(
      `.reduce-output-chart-${ppo}`
    );

    if (chartElement === null) {
      const plotWrapperElement = document.createElement('div');
      plotWrapperElement.classList.add('reduce-output-chart-wrapper');

      chartElement = document.createElement('canvas');
      chartElement.classList.add('reduce-output-chart');
      plotWrapperElement.appendChild(chartElement);

      document.querySelector('.reduce-output').appendChild(plotWrapperElement);
    }

    const data = { datasets: [] };

    for (const s of perSeriesOptions) {
      const dataPoints = [];

      for (const i of independentOptions) {
        for (const t of testCases) {
          if (
            t[perPlotParameter] == ppo &&
            t[perSeriesParameter] == s &&
            t[independentParameter] == i
          ) {
            dataPoints.push({
              x: t[independentParameter],
              y: t.execsPerSecond,
            });
          }
        }
      }

      data.datasets.push({
        label: s,
        data: dataPoints,
      });
    }

    new Chart(chartElement, {
      type: 'scatter',
      data: data,
    });
  }
}

export function clearOutputTable(): void {
  getTableBodyElement().innerHTML = '';
}

export function appendOutputTableRow(rowRecord: TableRowRecord): void {
  const rowElement = document.createElement('div');
  rowElement.setAttribute('role', 'row');

  rowElement.innerHTML = `
    <span role="cell">${rowRecord.algorithm}</span>
    <span role="cell">${rowRecord.numPoints}</span>
    <span role="cell">${rowRecord.workgroupSize}</span>
    <span role="cell">${rowRecord.numDispatches}</span>
    <span role="cell">${rowRecord.passed}</span>
  `;

  getTableBodyElement().appendChild(rowElement);
}

function appendOutputTableElement(topLevelElement: Element): void {
  const tableElement = document.createElement('div');
  tableElement.setAttribute('role', 'table');
  tableElement.classList.add('reduce-data-table');

  tableElement.innerHTML = `
    <div class="reduce-data-table-header" role="rowgroup">
      <div role="row">
        <span role="columnheader" aria-sort="none">Algorithm</span>
        <span role="columnheader" aria-sort="none"># Points</span>
        <span role="columnheader" aria-sort="none">Workgroup Size</span>
        <span role="columnheader" aria-sort="none"># Dispatches</span>
        <span role="columnheader" aria-sort="none">Pass</span>
      </div>
    </div>

    <div class="reduce-data-table-body" role="rowgroup" />
  `;

  topLevelElement.appendChild(tableElement);
}

/**
 * Adds a <style> element to <head> with page-specific styles. If the stylesheet
 * already exists, does nothing.
 */
function appendStylesheet(): void {
  const styleSheetId = 'reduce-styles';

  // Stylesheet already exists, don't add it again:
  if (document.querySelector(`#${styleSheetId}`) !== null) {
    return;
  }

  const styleSheet = document.createElement('style');
  styleSheet.setAttribute('id', styleSheetId);

  styleSheet.innerHTML = `
    .reduce-output {
      display: flex;
      margin-bottom: 2rem;
    }
    
    .reduce-output-chart-wrapper {
      display: flex;
      align-items: flex-end;
      flex: 30%;
      max-height: 420px;
    }
    
    .reduce-data-table {
      width: fit-content;
    }

    .reduce-data-table * {
      font-size: 14px;
    }

    .reduce-data-table [role="row"] {
      display: grid;
      grid-template-columns: 100px 90px 130px 100px 80px;
    }

    .reduce-data-table-header [role="columnheader"] {
      font-weight: bold;
      background-color: #282828;
      color: white;
      border-top: 1px solid #282828;
      border-right: 1px solid #282828;
    }

    .reduce-data-table-header [role="columnheader"]:first-of-type {
      border-left: 1px solid #282828;
    }

    .reduce-data-table [role="row"] > span {
      padding: 0.25rem;
    }

    .reduce-data-table-body {
      height: 400px;
      padding-right: 1rem;
      overflow: auto;
    }

    .reduce-data-table-body [role="cell"] {
      border-top: 1px solid grey;
      border-right: 1px solid grey;
    }

    .reduce-data-table-body [role="cell"]:first-of-type {
      border-left: 1px solid grey;
    }

    .reduce-data-table-body [role="row"]:last-of-type [role="cell"] {
      border-bottom: 1px solid grey;
    }`;

  document.head.appendChild(styleSheet);
}

function ensureTopLevelElement(parentNode: Node): Element {
  let topLevelElement = document.querySelector('.reduce-output');

  // If the top level element wasn't found, ensure we create it with the
  // correct class name and add it to the parent element:
  if (topLevelElement === null) {
    topLevelElement = document.createElement('div');
    topLevelElement.classList.add('reduce-output');

    parentNode.appendChild(topLevelElement);
  }

  return topLevelElement!;
}

function getTableBodyElement(): Element {
  const tableBodyElement = document.querySelector('.reduce-data-table-body');

  if (tableBodyElement === null) {
    throw new Error('Unable to find table body element');
  }

  return tableBodyElement;
}
