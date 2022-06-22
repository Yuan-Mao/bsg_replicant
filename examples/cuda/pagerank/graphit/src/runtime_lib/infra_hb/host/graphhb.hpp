#pragma once
#include <infra_gapbs/benchmark.h>
#include <infra_hb/host/device.hpp>
#include <infra_hb/host/vector.hpp>

#define CURRENT_POD SIM_CURRENT_POD
#define NUM_PODS 64

namespace hammerblade {
class GraphHB {
public:

	struct vertexdata {
		int32_t offset;
		int32_t degree;
	};

	GraphHB() {}

        GraphHB(Graph &&g) :
                _host_g(std::move(g))
                { init(); }

	~GraphHB() {}

        /* no copying */
	GraphHB(const GraphHB &other) = delete;

        GraphHB(GraphHB &&other) {
                moveFrom(other);
        }

        GraphHB &operator=(GraphHB &&other) {
                moveFrom(other);
                return *this;
        }

        decltype(Graph().num_nodes()) num_nodes() const {
                return _host_g.num_nodes();
        }

        decltype(Graph().num_nodes()) num_edges() const {
                return _host_g.num_edges();
        }

        using Vec = Vector<int32_t>;
        decltype(Vec().getBase()) getOutIndicesAddr() const {
                return _out_index.getBase();
        }
        decltype(Vec().getBase()) getOutNeighborsAddr() const {
                return _out_neighbors.getBase();
        }

        decltype(Vector<vertexdata>().getBase()) getOutVertexlistAddr() const {
                return _out_vertexlist.getBase();
        }

        decltype(Vec().getBase()) getInIndicesAddr() const {
                return _in_index.getBase();
        }
        
        decltype(Vec().getBase()) getInBlockDCSRIndicesAddr() const {
                return _in_block_dcsr_index.getBase();
        }

        decltype(Vector<uint32_t>().getBase()) getBoolTagsAddr() const {
                return _bool_tags.getBase();
        }

        decltype(Vec().getBase()) getInBlockIndicesAddr() const {
                return _in_block_index.getBase();
        }

        decltype(Vec().getBase()) getInC2SRNeighborsAddr() const {
                return _in_c2sr_neighbors.getBase();
        }

        decltype(Vector<float>().getBase()) getInC2SRValsAddr() const {
                return _in_c2sr_vals.getBase();
        }

        int getInDCSRIndicesLength() const {
                return _in_block_dcsr_index.getLength();
        }
       
        Vec getInIndices() const {
                return _in_index;
        }
        Vec getInNeighbors() const {
                return _in_neighbors;
        }
        Vec getInBlockIndices() const {
                return _in_block_index;
        }
        decltype(Vec().getBase()) getInNeighborsAddr() const {
                return _in_neighbors.getBase();
        }

        decltype(Vector<vertexdata>().getBase()) getInVertexlistAddr() const {
                return _in_vertexlist.getBase();
        }

        std::vector<int32_t> get_out_degrees() const {
                std::vector<int32_t> out_degrees (_host_g.num_nodes(), 0);
                for (NodeID n=0; n < _host_g.num_nodes(); n++){
        		out_degrees[n] = _host_g.out_degree(n);
                }
                return out_degrees;
        }

        std::vector<int32_t> get_in_degrees() const {
                std::vector<int32_t> in_degrees (_host_g.num_nodes(), 0);
                for (NodeID n=0; n < _host_g.num_nodes(); n++){
        		in_degrees[n] = _host_g.in_degree(n);
                }
                return in_degrees;
        }

        Graph & getHostGraph() {
                return _host_g;
        }

        int64_t out_degree(int v) const {
                return _host_g.out_degree(v);
        }

        int64_t in_degree(int v) const {
                return _host_g.in_degree(v);
        }

        int64_t calculate_c2sr_num(int32_t * index, int64_t V) {
          int64_t max[VCACHE_BANKS];
          for(int i = 0; i < VCACHE_BANKS; i++) {
            max[i] = 0;
          }
          for(int i = 0; i < V; i = i + VCACHE_BANKS) {
            int end = (i + VCACHE_BANKS) > V ? V : (i + VCACHE_BANKS);
            for(int j = i; j < end; j++) {
              max[j % VCACHE_BANKS] += (int64_t)(index[j+1] - index[j]); 
            }
          }
          int64_t max_val = 0;
          for(int i = 0; i < VCACHE_BANKS; i++) {
            if(max_val < max[i]) {
              max_val = max[i];
            }
          }
          int64_t div = max_val % CACHE_LINE == 0 ? (max_val / CACHE_LINE) : (max_val / CACHE_LINE + 1);
          return div * CACHE_LINE * VCACHE_BANKS; 
        }

        void calculate_blocked_index(uint32_t * bool_tag, int * block_index, int * index, int * in_neighbor, int64_t start, int64_t end, int64_t V) {
          int64_t rows_within_block = (V % NUM_PODS) == 0 ? (V / NUM_PODS) : (V / NUM_PODS + 1);
	  for(int i = start; i < end; i++){
            std::cout << "For row " << i << std::endl;
            int point[NUM_PODS];
            for(int k = 0; k < NUM_PODS; k++) {
              point[k] = 0;
            }
            for(int j = index[i]; j < index[i+1]; j++) {
              int idx = in_neighbor[j] / (int)rows_within_block;
//              std::cout << "in_neighbor[" << j << "] is " << in_neighbor[j] << ", and idx is " << idx << std::endl;
              point[idx]++;
            }
//            for(int p = 0; p < NUM_PODS; p++) {
//              std::cout << "point[" << p << "] is " << point[p] << std::endl;
//            }
            
            for(int k = 0; k < NUM_PODS+1; k++){
              if (k == 0) {
                block_index[(i-start) * (NUM_PODS+1) + k] = index[i];
              } else {
                block_index[(i-start) * (NUM_PODS+1) + k] = index[i];
                for(int j = 1; j <= k; j++) {
                  block_index[(i-start) * (NUM_PODS+1) + k] += point[j-1];
                }
              }             
            }
            bool temp_tag[NUM_PODS];
            for(int k = 0; k < NUM_PODS; k++) {
              if(block_index[(i-start) * NUM_PODS + k] != block_index[(i-start) * NUM_PODS + k+1]) {
                temp_tag[k] = true;
              } else {
                temp_tag[k] = false;
              }
              std::cout << temp_tag[k] << std::endl;
            }
            
            int num = NUM_PODS / 32;
            uint32_t temp_val[num];
            for(int k = 0; k < num; k++) {
              temp_val[k] = 0;
              for(int j = 0; j < 32; j++) {
                temp_val[k] = temp_val[k] << 1;
                temp_val[k] |= (uint32_t)(temp_tag[32*k+j] ? 1 : 0);
              }
              std::cout << std::hex << temp_val[k] << std::endl;
              bool_tag[(i - start) * num + k] = temp_val[k];              
            }
          } 
        }
  
        int64_t calculate_dcsr_index(int * block_dcsr_index, int * index, int pod_start, int pod_end) {
          int idx = 0;
          for(int i = pod_start; i < pod_end; i++) {
            if(index[i] != index[i+1]) {
              block_dcsr_index[idx] = i;
              idx++;
            }
          }
          return (int64_t) idx;
        }

private:

	static const hb_mc_eva_t DEVICE_NULLPTR = 0;

	void init() { initGraphOnDevice(); }

	void initGraphOnDevice() {

          if (true) {
            //throw hammerblade::runtime_error("transpose not supported");
            // convert
            std::vector<int32_t> index(num_nodes() + 1);
            int64_t rows_within_block = (num_nodes() % NUM_PODS) == 0 ? (num_nodes() / NUM_PODS) : (num_nodes() / NUM_PODS + 1);
            std::cout << "Simulating current pod " << CURRENT_POD << "with total nodes " << num_nodes() << " and " << rows_within_block << " rows within each pod under blocking partitioning" << std::endl;
            int64_t pod_row_start = CURRENT_POD * rows_within_block;   
            int64_t pod_row_end = (pod_row_start + rows_within_block) > num_nodes() ? num_nodes() : (pod_row_start + rows_within_block);
            int64_t length = pod_row_end - pod_row_start;
            std::vector<int32_t> block_dcsr_index(length);
//            std::vector<int32_t> blocked_index(length * (NUM_PODS+1));
//            std::vector<uint32_t> bool_tag(length * NUM_PODS / 32);
            std::vector<int32_t> tmp_deg = this->get_in_degrees();
	    std::vector<vertexdata> tmp_vertexlist(num_nodes());
	    # pragma omp parallel for
            for (int64_t i = 0; i < num_nodes(); i++) {
               index[i] = _host_g.in_index_shared_.get()[i] - _host_g.in_neighbors_shared_.get();
	       vertexdata tmp_elem = {.offset = index[i], .degree = tmp_deg[i]};
	       tmp_vertexlist[i] = tmp_elem;
            }
	    index[num_nodes()] = num_edges();
            for(int64_t i = 0; i < num_nodes() + 1; i++) {
              c2sr_index[i] = index[i];
            }
            int64_t offset = num_nodes() + 1;
            int64_t slotbuffer[VCACHE_BANKS];
            for(int64_t i = 0; i < VCACHE_BANKS; i++) {
              slotbuffer[i] = 0;
            }
            for(int64_t i = 0; i < num_nodes(); i++) {
              int64_t idx = i % VCACHE_BANKS;
              slotbuffer[idx] += (c2sr_index[i+1] - c2sr_index[i]);
              if(i < VCACHE_BANKS) {
                c2sr_index[offset + i] = 0;
              } 
              if (i + VCACHE_BANKS < num_nodes()) {
                c2sr_index[offset + i + VCACHE_BANKS] = slotbuffer[idx];
              }
            }
//	    this->calculate_blocked_index(bool_tag.data(), blocked_index.data(), index.data(), _host_g.in_neighbors_shared_.get(), pod_row_start, pod_row_end, num_nodes());
//            for(int i = 0; i < length * (NUM_PODS+1); i++) {
//              std::cout << blocked_index[i] << std::endl;
//            }
            int64_t b_dcsr_length = this->calculate_dcsr_index(block_dcsr_index.data(), index.data(), pod_row_start, pod_row_end);
            float percent = b_dcsr_length / length; 
            std::cout << "Non-zero rows " << b_dcsr_length << " total row " << length << std::endl;
//            int64_t c2sr_num = this->calculate_c2sr_num(index.data(), num_nodes());
//            for(int i = 0; i < b_dcsr_length; i++) {
//              std::cout << std::dec << block_dcsr_index[i] << std::endl;
//            }
//            for(int i = 0; i < length * NUM_PODS / 32; i=i+NUM_PODS/32) {
//              std::cout << "Bool tags of pod " << (i / (NUM_PODS/32)) << std::endl;
//              for(int j = 0; j < NUM_PODS / 32; j++) {
//                std::cout << std::hex << bool_tag[i + j];
//              }
//              std::cout << "\n";
//            }
            // allocate
	    _in_index = Vec(num_nodes() + 1);
	    _in_neighbors = Vec(num_edges());
//            _in_block_index = Vec(length * (NUM_PODS+1));
            _in_block_dcsr_index = Vec(b_dcsr_length);
//            _bool_tags = Vector<uint32_t>(length * NUM_PODS);
//	    _in_vertexlist = Vector<vertexdata>(num_nodes());
//            std::cout << "malloc c2sr_index" << c2sr_num << " elements" << std::endl;
//            _in_c2sr_index = Vec(c2sr_idxnum);
//            _in_c2sr_neighbors = Vec(c2sr_num);
//            _in_c2sr_vals = Vector<float>(c2sr_num);
//            std::cout << std::hex << this->getInIndicesAddr() << std::endl;
//            std::cout << std::hex << this->getInC2SRIndicesAddr() << std::endl;
	    // copy
	    _in_index.copyToDevice(index.data(), index.size());
            _in_block_dcsr_index.copyToDevice(block_dcsr_index.data(), b_dcsr_length);
//            _bool_tags.copyToDevice(bool_tag.data(), bool_tag.size());
//            _in_block_index.copyToDevice(blocked_index.data(), blocked_index.size());
	    _in_neighbors.copyToDevice(_host_g.in_neighbors_shared_.get(), num_edges());
//	    _in_vertexlist.copyToDevice(tmp_vertexlist.data(), tmp_vertexlist.size());
	  }

	  // out neighbor
	  std::vector<int32_t> index(num_nodes() + 1);
	  std::vector<int32_t> tmp_deg = this->get_out_degrees();
	  std::vector<vertexdata> tmp_vertexlist(num_nodes());
	  #pragma omp parallel for
	  for (int64_t i = 0; i < num_nodes(); i++) {
	    index[i] = _host_g.out_index_shared_.get()[i] - _host_g.out_neighbors_shared_.get();
	    vertexdata tmp_elem = {.offset = index[i], .degree = tmp_deg[i]};
	    tmp_vertexlist[i] = tmp_elem;
	  }
	  index[num_nodes()] = num_edges();
	  //allocate
//	  _out_index = Vec(num_nodes() + 1);
//	  _out_neighbors = Vec(num_edges());
//	  _out_vertexlist = Vector<vertexdata>(num_nodes());

	  //copy
//	  _out_index.copyToDevice(index.data(), index.size());
//	  _out_neighbors.copyToDevice(_host_g.out_neighbors_shared_.get(), num_edges());
//	  _out_vertexlist.copyToDevice(tmp_vertexlist.data(), tmp_vertexlist.size());
	}

	void exit() { freeGraphOnDevice(); }
        void freeGraphOnDevice() {}

	Graph _host_g;
        Vec   _out_index;
        Vec   _out_neighbors;
        Vec   _in_index;
        Vec   _in_neighbors;
        Vec   _in_block_dcsr_index;
        Vec   _in_block_index;
        Vec   _in_c2sr_neighbors;
        Vector<float>  _in_c2sr_vals;
        Vector<uint32_t> _bool_tags;
        Vector<vertexdata> _out_vertexlist;
	Vector<vertexdata> _in_vertexlist;

        void moveFrom(GraphHB & other) {
          _host_g = std::move(other._host_g);
          _out_index = std::move(other._out_index);
          _out_neighbors = std::move(other._out_neighbors);
          _in_index = std::move(other._in_index);
          _in_block_dcsr_index = std::move(other._in_block_dcsr_index);
          _in_block_index = std::move(other._in_block_index);
          _bool_tags = std::move(other._bool_tags);
          _in_c2sr_neighbors = std::move(other._in_c2sr_neighbors);
          _in_c2sr_vals = std::move(other._in_c2sr_vals);
          _in_neighbors = std::move(other._in_neighbors);
          _in_vertexlist = std::move(other._in_vertexlist);
          _out_vertexlist = std::move(other._out_vertexlist);
        }
};
}