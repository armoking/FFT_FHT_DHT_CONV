#include <bits/stdc++.h>
#define all(x) begin(x), end(x)

using namespace std;

mt19937 rnd(58); // random generator
const double PI = acos(-1);

vector<int> logs = {-1};

int Log2(int x) {
  while (logs.size() <= x) {
    logs.push_back(logs[logs.size() >> 1] + 1);
  }
  return logs[x];
}


vector<double> generateSequence(int n) {
  vector<double> result(n);
  for (int i = 0; i < n; i++) {
    result[i] = 1.0 / ((unsigned int)rnd() + 1) * rnd();
  }
  return result;
}

struct Statistics {
  size_t multOperationCounter = 0;
  size_t sumOperationCounter = 0;
};

auto calculateSimpleConvolution(const vector<double>& a, const vector<double>& b) {
  assert(a.size() == b.size());
  auto start = clock();
  Statistics stat;
  const int n = a.size();
  vector<double> result(a.size() + b.size() - 1);
  
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      result[i + j] += a[i] * b[j];
      stat.sumOperationCounter++;
      stat.multOperationCounter++;
    }
  }
  
  auto finish = clock();
  
  return tuple<double, Statistics, vector<double>>{(finish - start) * 1000.0 / CLOCKS_PER_SEC, stat, result};
}

struct Complex {
  double real, imag;
  Complex(double real = 0, double imag = 0) : real(real), imag(imag) {}

  Complex operator + (const Complex& o) const {
    return {real + o.real, imag + o.imag};
  }
  
  Complex operator - (const Complex& o) const {
    return {real + o.real, imag + o.imag};
  }
  
  Complex operator * (const Complex& o) const {
    return {real * o.real - imag * o.imag, real * o.imag + imag * o.real};
  }
  
  Complex operator / (const double& o) const {
    return {real / o, imag / o};
  }
  
};


auto precalc(int n) {
  vector<Complex> w(n);
  for (int i = 0; i < n; ++i) {
    w[i] = Complex(cos(i * PI / (n >> 1)), sin(i * PI / (n >> 1)));
  }
  return w;
}


void FFT(vector<Complex>& arr, int pwr, Statistics& stat, const vector<Complex>& w) {
  if (pwr == 0) return;
  
  const int size = 1 << pwr;
  
  vector<Complex> a(size >> 1), b(size >> 1);
  
  for (int i = 0; i < size; i++) {
    if (i & 1) {
      b[i >> 1] = arr[i];
    } else {
      a[i >> 1] = arr[i];
    }
  }
  
  FFT(a, pwr - 1, stat, w);
  FFT(b, pwr - 1, stat, w);
  
  for (int i = 0; i < size; i++) {  
    arr[i] = a[i % a.size()] + w[i << Log2(w.size() / arr.size())] * b[i % b.size()];
    stat.sumOperationCounter += 3;
    stat.multOperationCounter += 2;
  }
}


auto calculateConvolutionWithFFT(const vector<double>& a, const vector<double>& b) {
  Statistics stat;
  const int n = a.size();
  
  int power = 0;
  int sz1 = a.size();
  int sz2 = b.size();
  while ((1 << power) < (sz1 + sz2 - 1)) power++;
  int x = 1 << power;

  auto w = precalc(x);

  auto start = clock();
  
  vector<Complex> arr(x);
  vector<Complex> brr(x);
  for (int i = 0; i < n; i++) {
    arr[i] = {a[i], 0.0};
    brr[i] = {b[i], 0.0};
  }
  
  
  FFT(arr, power, stat, w);
  FFT(brr, power, stat, w);
  
  for (int i = 0; i < x; i++) {
    arr[i] = arr[i] * brr[i];
    stat.multOperationCounter += 2;
    stat.sumOperationCounter += 2;
  }
  
 
  FFT(arr, power, stat, w);
  vector<double> result(x);
  
  for (int i = 0; i < x; i++) {
    result[i] = arr[i].real / x;
    stat.multOperationCounter++;
  }
  reverse(begin(result) + 1, end(result));
  result.resize(a.size() + b.size() - 1);
  
  auto finish = clock();
  
  return tuple<double, Statistics, vector<double>>{(finish - start) * 1000.0 / CLOCKS_PER_SEC, stat, result};
}

void FHT(vector<double>& arr, Statistics& stat, const vector<double>& cosines, const vector<double>& sinuses) {
  if (arr.size() <= 1) return;
  const int n = arr.size();
  vector<double> a(n >> 1), b(n >> 1);
  for (int i = 0; i < n; i++) {
    if (i & 1) {
      b[i >> 1] = arr[i];
    } else {
      a[i >> 1] = arr[i];
    }
  }
  FHT(a, stat, cosines, sinuses);
  FHT(b, stat, cosines, sinuses);
  a.resize(n);
  b.resize(n);
  for (int i = 0; i < n / 2; i++) {
    a[i + n / 2] = a[i];
    b[i + n / 2] = b[i];
  }
  for (int k = 0; k < n; k++) {
    double c = cosines[k << Log2(cosines.size() / n)];
    double s = sinuses[k << Log2(cosines.size() / n)];
    arr[k] = a[k] + b[k] * c + b[(n - k) % n] * s; 
    stat.multOperationCounter += 2;
    stat.sumOperationCounter += 2;
  }
}

auto calculateConvolutionWithFHT(const vector<double>& a, const vector<double>& b) {
  Statistics stat;

  int power = 0;
  while ((1 << power) < (a.size() + b.size() - 1)) power++;
  int x = 1 << power;
  
  vector<double> cosines(x);
  vector<double> sinuses(x);
  for (int i = 0; i < x; i++) {
    cosines[i] = cos(2 * PI * i / x);
    sinuses[i] = sin(2 * PI * i / x);
  }
  auto start = clock();
  
  vector<double> arr(all(a));
  vector<double> brr(all(b));
  arr.resize(x);
  brr.resize(x);

  
  FHT(arr, stat, cosines, sinuses);
  FHT(brr, stat, cosines, sinuses);
  
  vector<double> result(x);
  
  for (int i = 0; i < x; i++) {
    result[i] = (
          arr[i] * brr[i]
        + arr[(x - i) % x] * brr[i]
        + arr[i] * brr[(x - i) % x]
        - arr[(x - i) % x] * brr[(x - i) % x]
      ) / 2;
    stat.multOperationCounter += 5;
    stat.sumOperationCounter += 3;
  }
  
  FHT(result, stat, cosines, sinuses);
  
  for (auto& value : result) {
    value /= x;
    stat.multOperationCounter++;
  }
  
  result.resize(a.size() + b.size() - 1);
  
  auto finish = clock();
  
  return tuple<double, Statistics, vector<double>>{(finish - start) * 1000.0 / CLOCKS_PER_SEC, stat, result};
}


void DHT(vector<double>& arr, Statistics& stat, const vector<double>& cosines, const vector<double>& sinuses) {
  const int n = arr.size();
  vector<double> result(n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      result[i] += arr[j] * (cosines[i * j % n] + sinuses[i * j % n]);
      stat.multOperationCounter += 1;
      stat.sumOperationCounter += 2;
    }
  }  
  arr = std::move(result);
}

auto calculateConvolutionWithDHT(const vector<double>& a, const vector<double>& b) {
  Statistics stat;

  int x = a.size() + b.size() - 1;
  
  vector<double> cosines(x);
  vector<double> sinuses(x);
  for (int i = 0; i < x; i++) {
    cosines[i] = cos(2 * PI * i / x);
    sinuses[i] = sin(2 * PI * i / x);
  }
  auto start = clock();
  
  vector<double> arr(all(a));
  vector<double> brr(all(b));
  arr.resize(x);
  brr.resize(x);

  
  DHT(arr, stat, cosines, sinuses);
  DHT(brr, stat, cosines, sinuses);
  
  vector<double> result(x);
  
  for (int i = 0; i < x; i++) {
    result[i] = (
          arr[i] * brr[i]
        + arr[(x - i) % x] * brr[i]
        + arr[i] * brr[(x - i) % x]
        - arr[(x - i) % x] * brr[(x - i) % x]
      ) / 2;
    stat.multOperationCounter += 5;
    stat.sumOperationCounter += 3;
  }
  
  DHT(result, stat, cosines, sinuses);
  
  for (auto& value : result) {
    value /= x;
    stat.multOperationCounter++;
  }
  
  result.resize(a.size() + b.size() - 1);
  
  auto finish = clock();
  
  return tuple<double, Statistics, vector<double>>{(finish - start) * 1000.0 / CLOCKS_PER_SEC, stat, result};
}


double calculateDelta(vector<double>& a, vector<double>& b) {
  double answer = 0;
  for (int i = 0; i < int(a.size()); i++) {
    answer += abs(a[i] - b[i]);
  }
  return answer;
}

int main() {
  freopen("output.txt", "w", stdout);
  for (int n = 1000; n <= 10000; n += 1000) {
    auto a = generateSequence(n);
    auto b = generateSequence(n);
    
    cout << "Size = " << n << '\n';
    cout << '\n';
    cout << "simple:\n";
    auto [spentTimeSimple, statSimple, resultSimple] = calculateSimpleConvolution(a, b);
    cout << spentTimeSimple << " ms.\n";
    cout << "Mult operations: " << statSimple.multOperationCounter << '\n';
    cout << "Sum operations: " << statSimple.sumOperationCounter << '\n';
    cout << '\n';
    
    cout << "fft:\n";
    auto [spentTimeFFT, statFFT, resultFFT] = calculateConvolutionWithFFT(a, b); 
    cout << spentTimeFFT << " ms.\n";
    cout << "Mult operations: " << statFFT.multOperationCounter << '\n';
    cout << "Sum operations: " << statFFT.sumOperationCounter << '\n';
    
    cout << "Differ fft - simple: " << calculateDelta(resultSimple, resultFFT) << '\n';
    cout << '\n';

    cout << "fht:\n";
    auto [spentTimeFHT, statFHT, resultFHT] = calculateConvolutionWithFHT(a, b); 
    cout << spentTimeFHT << " ms.\n";
    cout << "Mult operations: " << statFHT.multOperationCounter << '\n';
    cout << "Sum operations: " << statFHT.sumOperationCounter << '\n';
    
    cout << "Differ fht - simple: " << calculateDelta(resultSimple, resultFHT) << '\n';
    cout << '\n';
    
    cout << "dht:\n";
    auto [spentTimeDHT, statDHT, resultDHT] = calculateConvolutionWithDHT(a, b); 
    cout << spentTimeDHT << " ms.\n";
    cout << "Mult operations: " << statDHT.multOperationCounter << '\n';
    cout << "Sum operations: " << statDHT.sumOperationCounter << '\n';
    
    cout << "Differ dht - simple: " << calculateDelta(resultSimple, resultDHT) << '\n';
    cout << '\n';
  }
}
