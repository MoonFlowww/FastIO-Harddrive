#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>
#include "latte.hpp"


static constexpr int N = 5'000'000;

void write(){
  Latte::Mid::Start("MakeFields");
  auto model = ROOT::RNTupleModel::Create(); // create the var
  auto pt    = model->MakeField<float>("pt"); // MakeField create a filed aka approximatly "column"
  auto eta = model->MakeField<float>("eta");
  auto phi = model->MakeField<float>("phi");
  auto hits  = model->MakeField<std::vector<int>>("hits");
  auto energy = model->MakeField<double>("energy");
  Latte::Mid::Stop("MakeFields");

  ROOT::RNTupleWriteOptions opts;
  opts.SetCompression(101); // LZ4

  auto writer = ROOT::RNTupleWriter::Recreate(
    std::move(model), "Events", "./events.root", opts
  );


  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.f, 100.f);


  for(int i = 0; i<N; ++i){
    *pt     = dist(rng);
    *eta    = dist(rng) * 0.05f - 2.5f;
    *phi    = dist(rng) * 0.063f - 3.14f;
    *energy = dist(rng) * 10.0;
    *hits   = { i % 10, i % 7, i % 3 };
    writer->Fill();

    LATTE_PULSE("Gen");
  }
}



void read(){
  auto reader = ROOT::RNTupleReader::Open("Events", "events.root");

  //GetView is Lazy, think DeferredScalar
  auto vPt = reader->GetView<float>("pt");
  auto vEta = reader->GetView<float>("eta");
  auto vPhi = reader->GetView<float>("phi");
  auto vE = reader->GetView<float>("energy");

  //--------------------ALL---------------------
  double sum =0;
  for(auto i : reader->GetEntryRange()){ // return a range [0, N-1]
    sum+= vPt(i) + vEta(i) + vPhi(i) + vE(i);
    LATTE_PULSE("Sum_All");
  }
  std::cout << "Sum all: " << sum << std::endl;


  //------------------1Column--------------------
  sum=0;
  Latte::Mid::Start("Sum_One");
  for(auto i : reader->GetEntryRange()){ // return a range [0, N-1]
    sum+= vPt(i);
    LATTE_PULSE("Sum_One");
  }
  std::cout << "Sum (pt only): " << sum << std::endl;




  //----------------------Slice-------------------
  auto range = reader->GetNEntries(); // return uint64_t
  auto start = range*45/100; //from 45%
  auto stop  = range*55/100; // to 55%

  sum=0;
  for(uint64_t i = start; i<stop; ++i){
    sum += vPt(i);
    LATTE_PULSE("Sum_Slice");
  }
  std::cout << "Sum (pt Slice): " << sum << std::endl;
}



int main(){
  Latte::Hard::Start("Global");

  write();
  read();

  Latte::Hard::Stop("Global");

  Latte::DumpToStream(std::cout, Latte::Parameter::Time, Latte::Parameter::Raw);
}
