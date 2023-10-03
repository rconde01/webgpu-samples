import { makeSample, SampleInit } from '../../components/SampleLayout';
import Chart from 'chart.js/auto';

import reduceWGSL from './reduce.wgsl';

const init: SampleInit = async ({ canvas, pageState, gui }) => {
  const adapter = await navigator.gpu.requestAdapter();
  const device = (await adapter?.requestDevice({
    requiredLimits: {
      maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
      maxComputeInvocationsPerWorkgroup:
        adapter.limits.maxComputeInvocationsPerWorkgroup,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  })) as GPUDevice;

  if (!pageState.active) return;

  const topLevelDiv = document.createElement('div');
  topLevelDiv.style.display = 'flex';
  const logDiv = document.createElement('div');
  logDiv.style.flex = '30%';

  const logTable = document.createElement('table');

  const addRowToTable = (algorithm, points, workgroup, dispatches, pass) => {
    const row = document.createElement('tr');

    const cols: string[] = [];
    cols.push(`${algorithm}`);
    cols.push(`${points}`);
    cols.push(`${workgroup}`);
    cols.push(`${dispatches}`);
    cols.push(`${pass}`);

    for (const c of cols) {
      const td = document.createElement('td');
      td.textContent = c;
      row.appendChild(td);
    }

    logTable.appendChild(row);
  };

  addRowToTable('algorithm', 'points', 'workgroup', 'dispatches', 'pass');

  logDiv.appendChild(logTable);

  const plotDiv = document.createElement('div');
  plotDiv.style.flex = '30%';

  topLevelDiv.appendChild(logDiv);
  topLevelDiv.appendChild(plotDiv);

  if (canvas.parentNode) {
    canvas.parentNode.appendChild(topLevelDiv);
  } else {
    console.error('canvas.parentNode is null');
  }

  canvas.hidden = true;

  interface TestCase {
    algorithm: string;
    numPoints: number;
    workgroupSize: number;
    numDispatches: number;
    data: Float32Array;
    execsPerSecond: number;
  }

  const testCases: TestCase[] = [];
  let testDataBuffer: GPUBuffer;

  const vendor = (await adapter?.requestAdapterInfo())?.vendor;

  const maxWorkgroupSize = device.limits.maxComputeInvocationsPerWorkgroup;
  const maxWorkgroups = device.limits.maxComputeWorkgroupsPerDimension;
  const maxWorkgroupStorageSize = device.limits.maxComputeWorkgroupStorageSize;

  const workingBuffer0 = device.createBuffer({
    label: 'workingBuffer0',
    size: maxWorkgroups * maxWorkgroupSize * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const workingBuffer1 = device.createBuffer({
    label: 'workingBuffer1',
    size:
      (maxWorkgroups * maxWorkgroupSize * Float32Array.BYTES_PER_ELEMENT) / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const reduceShader = device.createShaderModule({
    label: 'reduceShader',
    code: reduceWGSL,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        // global input
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'read-only-storage',
        },
      },
      {
        // global output
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'storage',
        },
      },
    ],
  });

  const computePipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
    label: 'computePipelineLayout',
  });

  const numPointsOptions: number[] = [
    1_024, 16_384, 131_072, 524_288, 4_194_304,
  ];

  for (const numPoints of numPointsOptions) {
    const workgroupSizeOptions = getWorkgroupSizeOptions(
      vendor,
      maxWorkgroupSize,
      maxWorkgroups,
      maxWorkgroupStorageSize,
      Float32Array.BYTES_PER_ELEMENT,
      numPoints
    );

    for (const workgroupSize of workgroupSizeOptions) {
      for (const algoNumber of [0, 1, 2, 3]) {
        testCases.push({
          algorithm: 'reduce' + algoNumber.toString(),
          numPoints: numPoints,
          workgroupSize: workgroupSize,
          data: null,
          numDispatches: 0,
          execsPerSecond: 0,
        });
      }
    }
  }

  function reduceCpu(input: Float32Array): number {
    // TODO: Use Kahan summation?
    let result = 0;

    for (let i = 0; i < input.length; ++i) {
      result += input[i];
    }

    return result;
  }

  function getWorkgroupSizeOptions(
    vendor: string,
    maxWorkgroupSize: number,
    maxWorkgroups: number,
    maxWorkgroupStorageSize: number,
    bytesPerWorkgroupElement: number,
    numInputPoints: number
  ): number[] {
    const workgroupSizeOptions: number[] = [];

    let subgroupSize = 32;

    const vendorLowerCase = vendor.toLowerCase();

    // This is just a guess based on typical cases - these could be wrong in general
    // or drivers might dynamically change them based on a variety of factors.
    if (vendorLowerCase.includes('nvidia')) {
      subgroupSize = 32;
    } else if (vendorLowerCase.includes('intel')) {
      subgroupSize = 32;
    } else if (vendorLowerCase.includes('apple')) {
      subgroupSize = 32;
    } else if (
      vendorLowerCase.includes('amd') ||
      vendorLowerCase.includes('ati')
    ) {
      subgroupSize = 64;
    }

    const maxInvocationsPerDispatch = maxWorkgroupSize * maxWorkgroups;

    // We assume the entire input array can be reduced once in a single
    // dispatch.
    if (numInputPoints > maxInvocationsPerDispatch) {
      return workgroupSizeOptions;
    }

    let workgroupSize = maxWorkgroupSize;

    while (workgroupSize >= subgroupSize) {
      const firstNumGroups = Math.ceil(numInputPoints / workgroupSize);

      if (
        firstNumGroups <= maxWorkgroups &&
        workgroupSize * bytesPerWorkgroupElement <= maxWorkgroupStorageSize
      ) {
        workgroupSizeOptions.push(workgroupSize);
      }

      workgroupSize /= 2;
    }

    return workgroupSizeOptions;
  }

  const reduceGpu = (testCase: TestCase, compareAgainstCpu: boolean) => {
    const commandEncoder = device.createCommandEncoder({
      label: 'commandEncoder',
    });

    const computePass = commandEncoder.beginComputePass({
      label: 'computePass',
    });

    computePass.setPipeline(computePipeline);

    // Note: original CUDA does a copy between dispatches instead of swapping
    // buffers...in WebGPU you can't do that, you need multiple passes. I'm assuming
    // swapping buffers is faster but i don't know for sure.

    const getInputBuffer = (dispatchNum: number): GPUBuffer => {
      if (dispatchNum == 0) {
        return testDataBuffer;
      } else {
        if (dispatchNum % 2 == 1) {
          return workingBuffer0;
        } else {
          return workingBuffer1;
        }
      }
    };

    const getOutputBuffer = (dispatchNum: number): GPUBuffer => {
      if (dispatchNum % 2 == 0) {
        return workingBuffer0;
      } else {
        return workingBuffer1;
      }
    };

    // Keep reducing until dataSize is 1
    let dataSize = testCase.numPoints;
    let dispatchNum = 0;
    while (dataSize > 1) {
      const inputBuffer = getInputBuffer(dispatchNum);
      const outputBuffer = getOutputBuffer(dispatchNum);

      computePass.setBindGroup(
        0,
        device.createBindGroup({
          label: 'bindGroup0',
          layout: bindGroupLayout,
          entries: [
            {
              binding: 0,
              resource: {
                label: 'globalInput',
                buffer: inputBuffer,
                size: dataSize * Float32Array.BYTES_PER_ELEMENT,
              },
            },
            {
              binding: 1,
              resource: {
                label: 'globalOutput',
                buffer: outputBuffer,
              },
            },
          ],
        })
      );

      const numWorkgroups = Math.ceil(dataSize / testCase.workgroupSize);

      computePass.dispatchWorkgroups(numWorkgroups);

      dispatchNum++;
      dataSize = numWorkgroups;
    }

    computePass.end();

    device.queue.submit([commandEncoder.finish()]);

    if (compareAgainstCpu) {
      const transferCommandEncoder = device.createCommandEncoder({
        label: 'transferCommandEncoder',
      });

      const transferBuffer = device.createBuffer({
        label: 'transferBuffer',
        size: Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      transferCommandEncoder.copyBufferToBuffer(
        getOutputBuffer(dispatchNum - 1),
        0,
        transferBuffer,
        0,
        Float32Array.BYTES_PER_ELEMENT
      );

      device.queue.submit([transferCommandEncoder.finish()]);

      transferBuffer.mapAsync(GPUMapMode.READ).then(() => {
        const cpuResult = reduceCpu(testCase.data);

        const mappedBuffer = transferBuffer.getMappedRange(
          0,
          Float32Array.BYTES_PER_ELEMENT
        );

        const gpuResult = new Float32Array(mappedBuffer)[0];
        const scaledDiff = (cpuResult - gpuResult) / cpuResult;

        const passed = scaledDiff < 1e-6;

        addRowToTable(
          testCase.algorithm,
          testCase.numPoints,
          testCase.workgroupSize,
          testCase.numDispatches,
          passed
        );

        transferBuffer.unmap();
        transferBuffer.destroy();
      });
    }
  };

  let computePipeline: GPUComputePipeline;

  const initTestCase = (testCase: TestCase) => {
    testCase.data = new Float32Array(testCase.numPoints);
    testCase.numDispatches = calculateNumDispatches(testCases[testIndex]);

    for (let i = 0; i < testCase.numPoints; ++i) {
      testCase.data[i] = Math.random();
    }

    if (testDataBuffer !== undefined) {
      testDataBuffer.destroy();
    }

    testDataBuffer = device.createBuffer({
      label: 'testDataBuffer',
      size: testCase.numPoints * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });

    // Upload test data
    {
      const mappedTestBuffer = new Float32Array(
        testDataBuffer.getMappedRange()
      );
      mappedTestBuffer.set(testCase.data);
      testDataBuffer.unmap();
    }

    computePipeline = device.createComputePipeline({
      label: 'computePipeline',
      layout: computePipelineLayout,
      compute: {
        module: reduceShader,
        entryPoint: testCase.algorithm,
        constants: {
          workgroup_size: testCase.workgroupSize,
        },
      },
    });
  };

  function calculateNumDispatches(testCase: TestCase): number {
    let dataSize = testCase.numPoints;

    let numDispatches = 0;

    while (dataSize > 1) {
      const numWorkgroups = Math.ceil(dataSize / testCase.workgroupSize);
      dataSize = numWorkgroups;

      numDispatches++;
    }

    return numDispatches;
  }

  const sendExecuteMessage = () => {
    postMessagePending++;
    window.postMessage('execute', '*');
  };

  enum State {
    stopped,
    runningChecks,
    runningBenchmarks,
    stopRequested,
  }

  let state: State = State.stopped;

  let postMessagePending = 0;

  const casesPerPost = 100;
  const maxCaseCount = 100;
  let caseCount = 0;
  let testIndex = 0;
  let caseStartTime: number;

  // dimensions
  // * num points
  // * algorithm
  // * workgroupSize

  function removeDuplicates(arr) {
    let unique = [];
    arr.forEach((element) => {
      if (!unique.includes(element)) {
        unique.push(element);
      }
    });
    return unique;
  }

  const createPlot = (xAxisParameter, seriesParameter) => {};

  const createPlots = (xAxisParameter, seriesParameter) => {
    const parameters = ['algorithm', 'numPoints', 'workgroupSize'];

    const erase = (v) => {
      parameters.splice(parameters.indexOf(v), 1);
    };

    erase(xAxisParameter);
    erase(seriesParameter);

    const plotParameter = parameters[0];

    let plotVariants = [];

    for (const t of testCases) {
      plotVariants.push(t[plotParameter]);
    }

    plotVariants = removeDuplicates(plotVariants);
  };

  const reportBenchmarks = () => {
    const plot1Canvas = document.createElement('canvas');
    plotDiv.appendChild(plot1Canvas);

    new Chart(plot1Canvas, {
      type: 'bar',
      data: {
        labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
        datasets: [
          {
            label: '# of Votes',
            data: [12, 19, 3, 5, 2, 3],
            borderWidth: 1,
          },
        ],
      },
      options: {
        scales: {
          y: {
            beginAtZero: true,
          },
        },
      },
    });
  };

  const cleanupBenchmarks = () => {};

  const resetState = () => {
    caseCount = 0;
    testIndex = 0;
    state = State.stopped;
  };

  gui.add(
    {
      start: () => {
        if (state === State.stopped) {
          resetState();
          cleanupBenchmarks();
          state = State.runningChecks;
          sendExecuteMessage();
        }
      },
    },
    'start'
  );

  gui.add(
    {
      stop: () => {
        if (state !== State.stopped) {
          state = State.stopRequested;
        }
      },
    },
    'stop'
  );

  // Add case count control
  // Add selection for:
  // x-axis: [workgroupSize, algorithm, numpoints]
  // series: [workgroupSize, algorithm, numpoints]

  // For check phase report out results, passed, num dispatches

  window.onmessage = () => {
    postMessagePending--;

    if (postMessagePending == 0) {
      const testCase = testCases[testIndex];

      if (state === State.runningChecks) {
        if (testIndex < testCases.length) {
          initTestCase(testCase);
          reduceGpu(testCase, true);

          testIndex++;

          sendExecuteMessage();
        } else {
          state = State.runningBenchmarks;
          testIndex = 0;
          sendExecuteMessage();
        }
      } else if (state === State.runningBenchmarks) {
        if (caseCount === 0) {
          initTestCase(testCase);
          caseStartTime = Date.now();
        }

        for (let i = 0; i < casesPerPost; ++i) {
          reduceGpu(testCase, false);
        }

        caseCount += casesPerPost;

        if (caseCount >= maxCaseCount) {
          testCase.execsPerSecond = Math.floor(
            (1000.0 * maxCaseCount) / (Date.now() - caseStartTime)
          );

          testIndex++;

          if (testIndex < testCases.length) {
            // start the next test case
            caseCount = 0;
            sendExecuteMessage();
          } else {
            resetState();
            reportBenchmarks();
          }
        }
      } else if (state === State.stopRequested) {
        resetState();
        return;
      }
    }
  };
};

const Reduce: () => JSX.Element = () =>
  makeSample({
    name: 'Reduce',
    description:
      'Translates several CUDA reduce algorithms into WebGPU and benchmarks them with varying parameters.',
    gui: true,
    init,
    sources: [
      {
        name: __filename.substring(__dirname.length + 1),
        contents: __SOURCE__,
      },
      {
        name: 'reduce.wsgl',
        contents: reduceWGSL,
      },
    ],
    filename: __filename,
  });

export default Reduce;
