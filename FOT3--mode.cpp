
#include <iostream>
#include <vector>
#include <bitset>
#include <cmath>
#include <gmpxx.h>
#include <chrono> // For std::chrono::system_clock
#include <ctime>
#include <cstdlib>
#include <algorithm> // For std::shuffle
#include <random>
#include <utility>   // For std::pair
#include <unordered_map>

using namespace std;

// Custom hash function for mpz_class
struct mpz_class_hash {
    size_t operator()(const mpz_class& value) const {
        return hash<string>()(value.get_str()); // Hash the string representation of mpz_class
    }
};

// Find the most frequent message
mpz_class evaluate_mode(const std::vector<mpz_class>& messages) {
  std::unordered_map<mpz_class, int, mpz_class_hash> frequency_map; // Use custom hash function
  // Count the frequency of each message
  for (const auto& message : messages) {
    frequency_map[message]++;
  }
  // Find the most frequent message
  mpz_class most_frequent_message;
  int max_frequency = 0;
  for (const auto& [message, frequency] : frequency_map) {
    if (frequency > max_frequency) {
      most_frequent_message = message;
      max_frequency = frequency;
    }
  }
  return most_frequent_message;
}


//Padding function based on the padding technique from the preliminaries section of the paper
mpz_class add_padding(const mpz_class& message, int gamma, const mpz_class& modulus) {
  // Calculate 2^gamma using GMP's mpz_powm_ui for efficient exponentiation
  mpz_class two_pow_gamma;
  mpz_ui_pow_ui(two_pow_gamma.get_mpz_t(), 2, gamma); // Compute 2^gamma
  two_pow_gamma %= modulus; // Modulo reduction
  // Ensure that the modulus is large enough to accommodate the padding
  if (modulus <= two_pow_gamma) {
    throw std::runtime_error("Modulus is too small to handle the padding size.");
  }
  // Multiply the message by 2^gamma for padding
  mpz_class padded_message = (message * two_pow_gamma) % modulus;
  // Ensure the padded_message has gamma zero bits by verifying its lower bits
  if (padded_message % two_pow_gamma != 0) {
    throw std::runtime_error("Padding failed: lower bits are not zero after adding padding.");
  }
  return padded_message;
}

// // Function to verify and remove padding using the modular multiplicative inverse
mpz_class remove_padding(const mpz_class& padded_message, int gamma, const mpz_class& modulus) {
  // Check if the lower gamma bits are zero
  mpz_class gamma_check = padded_message % (mpz_class(1) << gamma);
  if (gamma_check != 0) {
    throw std::runtime_error("Padding check failed: lower bits are not zero.");
  }
  // Calculate 2^gamma
  mpz_class two_pow_gamma = mpz_class(1) << gamma;
  // Calculate the modular multiplicative inverse of 2^gamma mod modulus
  mpz_class gamma_inverse;
  if (mpz_invert(gamma_inverse.get_mpz_t(), two_pow_gamma.get_mpz_t(), modulus.get_mpz_t()) == 0) {
    throw std::runtime_error("Multiplicative inverse of 2^gamma mod modulus does not exist.");
  }
  // Remove padding using the modular multiplicative inverse
  mpz_class unpadded_message = (padded_message * gamma_inverse) % modulus;
  return unpadded_message;
}

// Generate a random number-- mainly for testing encode-evaluate-decode
mpz_class generate_random_(int bit_size) {
  static gmp_randclass rng(gmp_randinit_default); // Static to persist across calls
  static bool is_seeded = false;    // Tracks if seeding is done
  if (!is_seeded) {
    auto now = chrono::high_resolution_clock::now();
    auto seed = now.time_since_epoch().count(); // Use high-resolution clock for better randomness
    rng.seed(seed);          // Seed the generator
    is_seeded = true;
  }
  return rng.get_z_bits(bit_size); // Generate a random number
}

// Encode_mode function
vector<mpz_class> encode_mode(const vector<mpz_class>& messages, const mpz_class& modulus, int gamma, const vector<mpz_class>& random_values) {
  //vector<mpz_class> encoded_messages(messages.size());
  vector<mpz_class> encoded_messages;
  encoded_messages.reserve(messages.size());
  unordered_map<mpz_class, size_t, mpz_class_hash> unique_map; // Map to track unique padded messages and their indices in random_values
  if (messages.size() != random_values.size()) {
    throw runtime_error("The size of messages and random values must match.");
  }
  for (size_t i = 0; i < messages.size(); ++i) {
    // Step 1: Pad the message
    mpz_class padded_message = add_padding(messages[i], gamma, modulus);
    // Step 2: Check if the padded message is unique
    if (unique_map.find(padded_message) == unique_map.end()) {
      // Unique message: Use provided random value r_i
      mpz_class r_i = random_values[i];
      mpz_class h = (padded_message + r_i) % modulus;
      // Store index and encoded value
      encoded_messages.push_back(h);
      unique_map[padded_message] = i; // Map the padded message to its index
    } else {
      // Duplicate padded message: Use the random value from the previous occurrence
      size_t existing_index = unique_map[padded_message];
      mpz_class r_existing = random_values[existing_index];
      mpz_class h = (padded_message + r_existing) % modulus;
      encoded_messages.push_back(h);
    }
  }
  return encoded_messages;
}

// Decode_mode function
mpz_class Decode_mode(const mpz_class& encoded_value, const std::vector<mpz_class>& random_values, int gamma, const mpz_class& modulus) {
  mpz_class two_pow_gamma;
  mpz_ui_pow_ui(two_pow_gamma.get_mpz_t(), 2, gamma); // Compute 2^gamma
  for (const auto& r : random_values) {
    // Compute the additive inverse of r
    mpz_class additive_inverse = (modulus - r) % modulus;
    if (additive_inverse < 0) {
      additive_inverse += modulus;
    }
    // Add the additive inverse to the encoded value
    mpz_class potential_plaintext = (encoded_value + additive_inverse) % modulus;
    // Check if the lower gamma bits are zero
    mpz_class temp = potential_plaintext %two_pow_gamma;
    if (temp ==0){
      // If valid, remove the padding and return the result
      return remove_padding(potential_plaintext, gamma, modulus);
    }
  }
  throw std::runtime_error("Failed to decode: No valid padding found.");
}


// Phase 1: Setup--- it generates a vector of random values (for each invocation)
vector<vector<mpz_class>> Setup(int number_of_OT, int n, unsigned int bit_size, const mpz_class& modulus) {
  // Declare the outer vector to hold 'number_of_OT' vectors of random numbers
  vector<vector<mpz_class>> random_numbers_collection(number_of_OT, vector<mpz_class>(n));
  // Initialize GMP random state
  gmp_randclass rand_gen(gmp_randinit_default);
  rand_gen.seed(time(nullptr)); // Seed with current time
  // Generate 'number_of_OT' vectors of 'n' random numbers
  for (int j = 0; j < number_of_OT; ++j) {
    for (int i = 0; i < n; ++i) {
      random_numbers_collection[j][i]=rand_gen.get_z_bits(bit_size);
      random_numbers_collection[j][i]%= modulus;
    }
  }
  return random_numbers_collection;
}


// Function to generate a hash table. It will be used bu Find to fetch elements efficiently
unordered_map<int, int> createIndexMap(const vector<int>& v) {
  unordered_map<int, int> indexMap;
  for (int i = 0; i < v.size(); ++i) {
    indexMap[v[i]] = i;
  }
  return indexMap;
}

// Phase 2: genQuery---Query Generation-- It generates two vectors of queries: result[i], y[i] (for each invocation)
vector<vector<int>> genQuery(int number_of_OT, int p_size, vector<vector<int>> p, int n, vector<vector<int>>& y) {
  // Pre-allocate memory for the result and y vectors
  vector<vector<int>> result(number_of_OT);
  y.resize(number_of_OT);
  int t = p_size;
  // Generate the vector v with elements from 0 to n-1
  vector<int> v(n);
  for (int i = 0; i < n; ++i) {
    v[i] = i;
  }
  // Initialize random engine for shuffling
  random_device rd;
  default_random_engine rng(rd());
  for (int i = 0; i < number_of_OT; ++i) {
    // Shuffle v in-place
    shuffle(v.begin(), v.end(), rng);
    result[i] = v;
    // Directly use the shuffle result to assign y[i]
    y[i].resize(t);
    if (n == 2) {
      // Direct mapping for n = 2
      for (int j = 0; j < t; ++j) {
        y[i][j] = (p[i][j] == 0) ? v[0] : v[1];
      }
    } else {
      // General case for larger n
      unordered_map<int, int> map;
      map.reserve(n); // Reserve space to avoid rehashing
      for (int j = 0; j < n; ++j) {
        map[v[j]] = j;
      }
      for (int j = 0; j < t; ++j) {
        y[i][j] = map[p[i][j]];
      }
    }
  }
  return result;
}



vector<vector<mpz_class>> GenRes(const vector<vector<mpz_class>> m, int n, int number_of_OT, const vector<vector<mpz_class>> r, const vector<vector<int>>& w, int gamma, const mpz_class& modulus) {
  size_t m_size = n;  // Store m.size() in a variable to avoid recomputing
  vector<vector<mpz_class>> x(number_of_OT, vector<mpz_class>(m_size));
  vector<vector<mpz_class>> z(number_of_OT, vector<mpz_class>(m_size));
  // Preallocate memory for each vector in z using the stored m_size
  // Process each OT separately
  for (int j = 0; j < number_of_OT; ++j) {
    unordered_map<int, int> indexMap = createIndexMap(w[j]);
    vector<mpz_class> result = encode_mode(m[j], modulus, gamma, r[j]);
    for (size_t i = 0; i < m_size; ++i) {
      // Perform XOR operation
      //mpz_class xor_result = m[i] ^ r[j][i];
      //mpz_class result = encode(m[i], r[j][i], modulus);
      //mpz_class result = encode_mode(m[i], modulus, gamma, r[j][i]);

      //z[j].push_back(result[i]);
      //z[j][i] = result[i];
      // Find the corresponding index in x
      auto it = indexMap.find(i);
      if (it != indexMap.end()) {
        int y_i = it->second;
        //cout<<"\n in  GenRes--res["<<i<<"]: "<<result[i]<<endl;
        x[j][y_i] = result[i];
        //cout<<"\n in  GenRes--x["<<j<<"]["<<y_i<<"]"<<x[j][y_i]<<endl;
      } else {
        cerr << "\n ** Error: invalid index during computing GenRes" << endl;
        return {}; // Return an empty vector to indicate an error
      }
    }
  }
  //cout<<"\n **** in  GenRes--Random-check--x["<<1<<"]["<<1<<"]"<<x[1][1]<<endl;
  // for (int i=0; i<number_of_OT; i++){
  //   for(int j=0; j< m_size; j++){
  //     cout<<"\n x["<<i<<"]["<<j<<"]: "<<x[i][j]<<endl;
  //     }
  // }
  return x;
}



////////





// Phase 3: old---GenRes---Generates a response to the receiver's query (for each invocation)
vector<vector<mpz_class>> GenRes(const vector<mpz_class> m, int number_of_OT, const vector<vector<mpz_class>> r, const vector<vector<int>>& w, int gamma, const mpz_class& modulus) {
  size_t m_size = m.size();  // Store m.size() in a variable to avoid recomputing
  vector<vector<mpz_class>> x(number_of_OT, vector<mpz_class>(m_size));
  vector<vector<mpz_class>> z(number_of_OT, vector<mpz_class>(m_size));
  // Preallocate memory for each vector in z using the stored m_size
  for (int j = 0; j < number_of_OT; ++j) {
    z[j].reserve(m_size);
  }
  // Process each OT separately
  for (int j = 0; j < number_of_OT; ++j) {
    unordered_map<int, int> indexMap = createIndexMap(w[j]);
    vector<mpz_class> result = encode_mode(m, modulus, gamma, r[j]);
    for (size_t i = 0; i < m_size; ++i) {
      // Perform XOR operation
      //mpz_class xor_result = m[i] ^ r[j][i];
      //mpz_class result = encode(m[i], r[j][i], modulus);
      //mpz_class result = encode_mode(m[i], modulus, gamma, r[j][i]);

      //z[j].push_back(result[i]);
      //z[j][i] = result[i];
      // Find the corresponding index in x
      auto it = indexMap.find(i);
      if (it != indexMap.end()) {
        int y_i = it->second;
        //cout<<"\n in  GenRes--res["<<i<<"]: "<<result[i]<<endl;
        x[j][y_i] = result[i];
        //cout<<"\n in  GenRes--x["<<j<<"]["<<y_i<<"]"<<x[j][y_i]<<endl;
      } else {
        cerr << "\n ** Error: invalid index during computing GenRes" << endl;
        return {}; // Return an empty vector to indicate an error
      }
    }
  }
  //cout<<"\n **** in  GenRes--Random-check--x["<<1<<"]["<<1<<"]"<<x[1][1]<<endl;
  return x;
}

// Phase 4: oblFilter---Oblivious filtering--returns a single message (for each invocation)
vector<mpz_class> oblFilter(int number_of_OT, int p_size, const vector<vector<mpz_class>> res_s, const vector<vector<int>> y, const mpz_class modulus) {
  // Preallocate the outer vector with the correct size
  vector<mpz_class> temp_res(p_size);
  vector<mpz_class> res_final(number_of_OT);
  // Iterate over the number of OTs
  for (int j = 0; j < number_of_OT; ++j) {
    // Iterate over each element in the current OT
    for (int i = 0; i < p_size; ++i) {
      // Assign the value from res_s based on the index from y
      temp_res[i] = res_s[j][y[j][i]];
    }
    res_final[j] = evaluate_mode(temp_res);
  }
  return res_final;
}


// Phase 4: retreive---messgae retreival (for each invocation)
mpz_class retreive(const mpz_class val, const vector<mpz_class> r, const vector<int> p, int p_size, int gamma, const mpz_class modulus) {
  // Check if j is within the valid range

  // Check if r has enough elements
    if (p_size > r.size()) {
        cerr << "Error: Out-of-bounds access in retreive function. p_size=" << p_size << ", r.size()=" << r.size() << endl;
        return mpz_class(0);  // Return a default value to avoid a crash
    }

  vector<mpz_class> random_values;
  for(int i = 0;i < p_size; i++){
    if (p[i] >= r.size()) {  // Ensure p[i] is within range of r
            cerr << "Error: Index out of bounds in retreive function. p[" << i << "]=" << p[i] << ", r.size()=" << r.size() << endl;
            return mpz_class(0);  // Return a default value to avoid segmentation fault
        }

    random_values.push_back(r[p[i]]);
  }
  mpz_class res = Decode_mode(val, random_values, gamma, modulus);
  return res;
}


// Function to generate a random big integer with a specified number of bits-- used for test
mpz_class generate_random_bigint(gmp_randclass& rng, int num_bits) {
  return generate_random_(num_bits);
  //return rng.get_z_bits(num_bits);
}

// generates vectors of random values
vector<vector<int>> generateRandomVectors(int p_size, int number_of_OT, int n) {
  // Vector to hold the collection of vectors
  vector<vector<int>> random_vectors;
  random_vectors.reserve(number_of_OT); // Reserve space for efficiency
  // Generate 'number_of_OT' vectors
  for (int i = 0; i < number_of_OT; ++i) {
    // Create a vector with all possible values from 0 to n-1
    vector<int> values(n);
    for (int j = 0; j < n; ++j) {
      values[j] = j;
    }
    // Seed the random number generator with a unique seed
    unsigned seed = chrono::system_clock::now().time_since_epoch().count() + i;
    shuffle(values.begin(), values.end(), mt19937(seed));
    // Select the first 'p_size' elements to ensure they are distinct
    vector<int> vec(values.begin(), values.begin() + p_size);
    // Add the generated vector to the collection
    random_vectors.push_back(move(vec));
  }
  return random_vectors;
}


mpz_class generate_prime(int bit_size) {
    gmp_randclass rng(gmp_randinit_default);

    // Use high-resolution clock for better seeding
    auto now = std::chrono::high_resolution_clock::now();
    auto seed = now.time_since_epoch().count();
    rng.seed(seed);

    mpz_class prime;

    // Ensure the generated number has exactly `bit_size` bits
    do {
        prime = rng.get_z_bits(bit_size); // Generate a random number of `bit_size` bits
        // Ensure it is within the desired bit range by setting the most significant bit (MSB)
        mpz_class msb_mask = mpz_class(1) << (bit_size - 1);
        prime |= msb_mask;
        // Find the next prime greater than or equal to this number
        mpz_nextprime(prime.get_mpz_t(), prime.get_mpz_t());

    } while (mpz_sizeinbase(prime.get_mpz_t(), 2) != bit_size); // Ensure correct bit length

    return prime;
}


///////////////////////////////
int main() {
  int n = 16;
  // // 16, 2^8 = 256, 2^12 = 4096, 2^16 = 65536, 2^20 = 1048576, 2^24 = 16777216, 2^28 = 268435456
  int p_size = 12;// it is t in t-out-of-n OT
  int number_of_OT = 10;// 125000, 250000, 500000, 1250000
  int message_size = 128;
  int gamma = 128;
  int bit_size = message_size + gamma + 1;

  int number_of_tests = 1;
  float phase1_, phase2_, phase3_, phase4_, phase5_;
  float phase1_total, phase2_total, phase3_total, phase4_total, phase5_total;
  float counter = 0;
   mpz_class modulus = generate_prime(bit_size);
  //cout<<"\n======= Message ========"<< endl;
  for(int i = 0; i < number_of_tests; i++){
    vector<vector<int>>y;
    vector<vector<int>> collection_of_p = generateRandomVectors(p_size, number_of_OT, n);
    // ****----Start_1: uncomment the below lines to get the values of p (indices)
    // cout<<"\n========= P ======"<<endl;
    // for(int j=0;j<number_of_OT;j++){
    //   cout<<"\n"<<j<<"th-invocation of OT"<<endl;
    //   for (int i = 0; i < p_size; ++i) {
    //     cout<<"\n p["<<j<<"]["<<i<<"]: "<< collection_of_p[j][i]<<endl;
    //   }
    //    cout<<"\n...."<<endl;
    //  }
    // ----End_1
    gmp_randclass rng(gmp_randinit_default);
    rng.seed(time(nullptr));
    //vector<mpz_class> m;
    //generate n random messages
    vector<vector<mpz_class>> messages(number_of_OT, vector<mpz_class>(n));
    for(int j = 0; j<number_of_OT; j++){
      for (int i = 0; i < n; ++i) {
        messages[j][i]=generate_random_bigint(rng, message_size);
      }
    }
    //cout<<"\n modulous: "<<modulus<<endl;
    //cout<<"\n------"<<endl;

    // ****----Start_2: uncomment below lines to get the values of messages (held by the sender)
    // for (int i = 0; i < n; ++i) {
    // cout<<"\n m["<<i<<"]: "<< m[i]<<endl;
    // }
    if(p_size < 5){
      for (int i=0; i<number_of_OT; i++){
        messages[i][collection_of_p[i][0]] = messages[i][collection_of_p[i][1]];
      }
    }
    else{
      for (int i=0; i<number_of_OT; i++){
        messages[i][collection_of_p[i][0]] = messages[i][collection_of_p[i][1]];
        messages[i][collection_of_p[i][2]]= messages[i][collection_of_p[i][1]];
        messages[i][collection_of_p[i][3]]= messages[i][collection_of_p[i][4]];
      }
    }
    // uncomment below to print out the messages after duplication
    //   for (int j = 0; j < n; ++j) {
    //     cout<<"\n m["<<j<<"]: " << m[j]<<endl;
    // }

    // cout<<"\n\n ++++++++++++++"<< endl;
    // for (int i=0; i<number_of_OT; i++){
    //   for (int j = 0; j < n; ++j) {
    //     cout<<"\n messages["<<i<<"][" <<j<<"]: "<< messages[i][j]<<endl;
    //   }
    // }

    //cout<<"\n======SetuP========="<<endl;
    double phase1 = 0;//time related variable
    double start_phase1 = clock();//time related variable
    vector<vector<mpz_class>> r = Setup(number_of_OT, n, bit_size, modulus);
    double end_phase1 = clock();//time related variable
    phase1 = end_phase1 - start_phase1;//time related variable
    phase1_ = phase1 / (double) CLOCKS_PER_SEC;//time related variable
    //cout<<"\n======genQuery========="<<endl;
    double phase2 = 0;//time related variable
    double start_phase2 = clock();//time related variable
    //GenQuery
    vector<vector<int>> w = genQuery(number_of_OT, p_size, collection_of_p, n, y);
    double end_phase2 = clock();//time related variable
    phase2 = end_phase2 - start_phase2;//time related variable
    phase2_ = phase2 / (double) CLOCKS_PER_SEC;//time related variable
    // ----Start_3: uncomment below lines to get the values of queries
    // for(int k = 0;k<number_of_OT; k++){
    //   for (int i = 0; i < n; ++i) {
    //     cout<<"\n w: "<<w[k][i]<<endl;
    //   }
    //   cout<<"\n........."<<endl;
    //   for (int i = 0; i < p_size; ++i) {
    //     cout<<"\n y: "<<y[k][i]<<endl;
    //   }
    //   cout<<"\n---------"<<endl;
    // }
    // ----End_3
    //cout<<"\n======GenRes========="<<endl;
    double phase3 = 0;//time related variable
    double start_phase3 = clock();//time related variable
    // GenRes
    vector<vector<mpz_class>> res_s = GenRes(messages, n, number_of_OT, r, w, gamma, modulus);
    double end_phase3 = clock();//time related variable
    phase3 = end_phase3 - start_phase3;//time related variable
    phase3_ = phase3 / (double) CLOCKS_PER_SEC;//time related variable
    //cout<<"\n======oblFilter========="<<endl;
    double phase4 = 0;//time related variable
    double start_phase4 = clock();//time related variable
    // oblFilter
    vector<mpz_class> res_h(number_of_OT);
    res_h = oblFilter(number_of_OT, p_size, res_s, y, modulus);
    double end_phase4 = clock();//time related variable
    phase4 = end_phase4 - start_phase4;//time related variable
    phase4_ = phase4 / (double) CLOCKS_PER_SEC;//time related variable
    //cout<<"\n======retreive========="<<endl;
    double phase5 = 0;//time related variable
    double start_phase5 = clock();//time related variable

    // retreive (the final result(s))
    for(int k = 0; k < number_of_OT; k++){
      mpz_class retreived_message = retreive(res_h[k], r[k], collection_of_p[k],  p_size, gamma,  modulus);
      //Uncomment the blelow lines to print each plaintext retreived message
      //cout<<"\n retreived_message["<<k<<"]: "<<retreived_message<<endl;
      //cout<<"\n\n ++ retrived: "<<k<<endl;
      //}
      //cout<<"\n\n++++++++++++++"<<endl;
    }
    double end_phase5 = clock();//time related variable
    phase5 = end_phase5 - start_phase5;//time related variable
    phase5_ = phase5 / (double) CLOCKS_PER_SEC;//time related variable
    phase1_total += phase1_;
    phase2_total += phase2_;
    phase3_total += phase3_;
    phase4_total += phase4_;
    phase5_total += phase5_;
    counter += phase5_ + phase4_ + phase3_ + phase2_ + phase1_;
  }
  cout<<"\n\n number_of_OT: "<<number_of_OT<<endl;
  cout<<"\n\n================== Runtime Breakdown (in Second)========================"<<endl;
  cout<<"\n Phase 1 runtime: "<<phase1_total/number_of_tests<<endl;
  cout<<"\n Phase 2 runtime: "<<phase2_total/number_of_tests<<endl;
  cout<<"\n Phase 3 runtime: "<<phase3_total/number_of_tests<<endl;
  cout<<"\n Phase 4 runtime: "<<phase4_total/number_of_tests<<endl;
  cout<<"\n Phase 5 runtime: "<<phase5_total/number_of_tests<<endl;
  cout<<"\n\n================== Total Runtime for FOT--Mode ========================"<<endl;
  cout<<"\n Setting:  Priority OT, t= " <<p_size<<", n= "<<n<<endl;
  cout<<"\n Total cost for: " <<number_of_OT<<" OT executions: "<<counter/number_of_tests<<endl;
  cout<<"\n\n=======================================================\n"<<endl;
  cout<<"\n n: "<<n<<endl;

  cout<<"\n\n****************** MODE ******************************"<<endl;
  cout<<"\n\n************************************************"<<endl;
  cout<<"\n\n================== Runtime Breakdown (in MilliSecond)========================"<<endl;
  cout<<"\n Phase 1 runtime: "<<1000*(phase1_total/number_of_tests)<<endl;
  cout<<"\n Phase 2 runtime: "<<1000*(phase2_total/number_of_tests)<<endl;
  cout<<"\n Phase 3 runtime: "<<1000*(phase3_total/number_of_tests)<<endl;
  cout<<"\n Phase 4 runtime: "<<1000*(phase4_total/number_of_tests)<<endl;
  cout<<"\n Phase 5 runtime: "<<1000*(phase5_total/number_of_tests)<<endl;
  cout<<"\n\n================== Total Runtime for FOT--Mode========================"<<endl;
  cout<<"\n Setting:  Priority OT, t= " <<p_size<<", n= "<<n<<endl;
  cout<<"\n Total cost for: " <<number_of_OT<<" OT executions: "<<1000*(counter/number_of_tests)<<endl;
  cout<<"\n\n=======================================================\n"<<endl;

  return 0;
}


//// g++ -std=c++11 -lgmpxx -lgmp main.cpp -o main
