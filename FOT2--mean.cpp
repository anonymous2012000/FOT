
#include <helib/helib.h>
#include <iostream>
#include <vector>
#include <helib/debugging.h>  // Provides helib::to_ZZX
using helib::to_ZZX;
using namespace std;


int main() {
  // Parameters
  unsigned long p = 4999;
  unsigned long m = 32109;
  unsigned long r = 1;
  unsigned long bits = 300;
  unsigned long c = 2;
  int t = 3;// t in t-out-of-n for each vector. Total threshold: t*  loop_size
  int loop_size = 1;// loop_size * nslots = total number of messages
  helib::Context context = helib::ContextBuilder<helib::BGV>()
                                 .m(m)
                                 .p(p)
                                 .r(r)
                                 .bits(bits)
                                 .c(c)
                                 .build();
  cout << "Security: " << context.securityLevel() <<  endl;
  // Key generation
  helib::SecKey secret_key(context);
  secret_key.GenSecKey();
  helib::addSome1DMatrices(secret_key); // Needed for rotations
  const helib::PubKey& public_key = secret_key;
  const helib::EncryptedArray& ea = context.getEA();
  long nslots = ea.size();
  cout << "\n nslots: " << nslots <<  endl;
  helib::Ctxt encrypted_sum_(public_key);
  for (int k = 0; k<loop_size; k++){
    vector<long> indices(t);
    for(int i=0; i<t; i++){
      indices[i] = i+5 ;
    }
    // Sender's original messages
    vector<long> messages(nslots, 0);
    for(int i=0; i<nslots; i++){
      messages[i] = i+1;// just an arbitrary choice
      cout<<"\n\noriginal messages["<<i<<"]: "<<messages[i]<< endl;
    }
    cout<<"\n\n------------------\n\n";
    for(int j=0; j < t; j++){
      cout<<"\n\n prefferred index: "<<indices[j]<<", prefferred messages: "<<messages[indices[j]]<< endl;
    }
    cout<<"\n\n------------------\n\n";

    helib::Ctxt encrypted_sum(public_key);
    for (long idx : indices) {
      vector<long> selector(nslots, 0);
      selector[idx] = 1;
      helib::Ctxt encrypted_selector(public_key);
      ea.encrypt(encrypted_selector, public_key, selector);

      helib::Ptxt<helib::BGV> ptxt_messages(context, messages);
      encrypted_selector.multByConstant(ptxt_messages);
      encrypted_sum += encrypted_selector;
    }
    // Sum across slots
    for (long i = 1; i < nslots; i <<= 1){
      helib::Ctxt tmp = encrypted_sum;
      ea.rotate(tmp, i);
      encrypted_sum += tmp;
    }
    encrypted_sum_+=encrypted_sum;
  }
  // Decrypt the total sum
  helib::Ptxt<helib::BGV> decrypted(context);
  secret_key.Decrypt(decrypted, encrypted_sum_);
  // Decrypt the total sum
  // Each slot is a PolyMod, use explicit cast to get long
  cout<<"\n\n-----------Result-------\n\n";

  long total_sum = static_cast<long>(decrypted[0]);
  cout << "Decrypted sum: " << total_sum <<  endl;
  cout << "Mean: " << total_sum / (t*loop_size) <<  endl;
  cout<<"\n\n-----------Setting-------\n\n";
  cout<<"\n\n Total number of messegase: "<<nslots*loop_size<< endl;
  cout<<"\n\n Threshold: "<<t*loop_size << endl;
  cout<< endl<<endl;
  return 0;
}
