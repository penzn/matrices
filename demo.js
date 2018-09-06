function init(arr, size) {
  for (let i = 0; i < (size*size); ++i) {
    arr[i] = i % size;
  }
}

function verify(arr, size) {
  for (let i = 0; i < (size*size); ++i) {
    if (arr[i] !== i % size) {
      print("Index: " + i + ", value: " + arr[i]);
      return false;
    }
  }
  return true;
}

function print_4(arr, off) {
  print("2x2 data: ");
  print(arr[off]     + " " + arr[off + 1]);
  print(arr[off + 2] + " " + arr[off + 3]);
}

function print_16(arr) {
  print("4x4 data: ");
  print(arr[0] + " " + arr[1] + " " + arr[2] + " " + arr[3]);
  print(arr[4] + " " + arr[5] + " " + arr[6] + " " + arr[7]);
  print(arr[8] + " " + arr[9] + " " + arr[10] + " " + arr[11]);
  print(arr[12] + " " + arr[13] + " " + arr[14] + " " + arr[15]);
}

const N = 20000; // Matrix size
let pages = Math.ceil(N * N / 16384.0);
print("Need " + pages + " pages");
const memObj = new WebAssembly.Memory({initial:pages});
const module = new WebAssembly.Module(readbuffer('matrices.wasm'));
const instance = new WebAssembly.Instance(module, { "dummy" : { "memory" : memObj } }).exports;
let data = new Float32Array (memObj.buffer);

print("Matrix size is " + N);

init(data, N);

var tStart = Date.now();
instance["transpose_f32"](0, N);
instance["transpose_f32"](0, N);
var tEnd = Date.now();
print("Two transpositions took " + (tEnd - tStart) + " milliseconds.");

if (verify(data, N)) {
  print("Verification passed");
} else {
  print("Verification failed");
}
