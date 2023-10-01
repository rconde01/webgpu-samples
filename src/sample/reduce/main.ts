import test from 'node:test';
import { makeSample, SampleInit } from '../../components/SampleLayout';

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

  const reduceGpu = (
    testDataSize: number,
    workgroupSize: number,
    compareAgainstCpu: boolean,
    testDataCpu: Float32Array | null
  ) => {
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
    let dataSize = testDataSize;
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

      const numWorkgroups = Math.ceil(dataSize / workgroupSize);

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
        console.log('CPU Reduce: ' + reduceCpu(testDataCpu));

        const mappedBuffer = transferBuffer.getMappedRange(
          0,
          Float32Array.BYTES_PER_ELEMENT
        );

        console.log('GPU Reduce: ' + new Float32Array(mappedBuffer)[0]);

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
  const maxCaseCount = 10000;
  let caseCount = 0;
  let testIndex = 0;
  let caseStartTime: number;

  const reportBenchmarks = () => {};

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

  window.onmessage = () => {
    postMessagePending--;

    if (postMessagePending == 0) {
      const testCase = testCases[testIndex];

      if (state === State.runningChecks) {
        if (testIndex < testCases.length) {
          initTestCase(testCase);
          reduceGpu(
            testCase.numPoints,
            testCase.workgroupSize,
            true,
            testCase.data
          );

          testIndex++;

          sendExecuteMessage();
        } else {
          state = State.runningBenchmarks;
          sendExecuteMessage();
        }
      } else if (state === State.runningBenchmarks) {
        if (caseCount === 0) {
          initTestCase(testCase);
          caseStartTime = Date.now();
        }

        for (let i = 0; i < casesPerPost; ++i) {
          reduceGpu(testCase.numPoints, testCase.workgroupSize, false, null);
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
