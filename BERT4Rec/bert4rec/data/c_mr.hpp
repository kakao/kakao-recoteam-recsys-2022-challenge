#pragma once

#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <numeric>
#include <sstream>
#include <utility>
#include <iterator>
#include <algorithm>
#include <unordered_map>

static inline std::string __join(const std::vector<std::string>& v) {
    if (v.empty())
        return "";
    return std::accumulate(v.begin() + 1, v.end(), v[0], [](auto& ss, const auto& s) {
        return ss + ' ' + s;
    });
}


class CMapReducer {
 public:
    CMapReducer() = delete;
    CMapReducer(std::string fname, std::string fname_timestamp, std::string tmp_dir, std::string pretrained_key_fname,
                int frequency_min_count, int consume_min_count, int sequence_length,
                int max_samples_per_stream, int num_workers) : fname_(fname), fname_timestamp_(fname_timestamp), tmp_dir_(tmp_dir),
                pretrained_key_fname_(pretrained_key_fname), frequency_min_count_(frequency_min_count), consume_min_count_(consume_min_count),
                sequence_length_(sequence_length), max_samples_per_stream_(max_samples_per_stream), num_workers_(num_workers) {}

    void run() {
        __map();
        __reduce();
    }

 private:
    void __split_chunks() {
        std::ifstream fin, fin_timestamp;
        std::ofstream fout, fout_timestamp;
        std::string line, item, line_timestamp, item_timestamp;
        std::stringstream ss, ss_timestamp;

        // First, build item counts.
        std::unordered_map<std::string, int> counts;
        fin = std::ifstream(fname_);
        while (std::getline(fin, line)) {
            ss = std::stringstream(line);
            while (std::getline(ss, item, ' '))
                ++counts[item];
        }
        fin.close();

        // Second, load pretrained_keys if exists.
        bool use_pretrained_keys = !pretrained_key_fname_.empty();
        bool use_timestamp = !fname_timestamp_.empty();
        if (use_pretrained_keys) {
            fin = std::ifstream(pretrained_key_fname_);
            std::vector<std::string> keys{std::istream_iterator<std::string>(fin), std::istream_iterator<std::string>()};
            for (const auto& key : keys)
                idmap_[key] = static_cast<int>(idmap_.size());
            fin.close();
        }

        // Third, filter out items from stream.
        fin = std::ifstream(fname_);
        fout = std::ofstream(fname_ + ".tmp");
        if (use_timestamp){
            fin_timestamp = std::ifstream(fname_timestamp_);
            fout_timestamp = std::ofstream(fname_timestamp_ + ".tmp");
        }
        
        int offset;
        int num_lines = 0;
        while (std::getline(fin, line)) {
            ss = std::stringstream(line);
            std::vector<std::string> stream, stream_timestamp_origin, stream_timestamp;

            if (use_timestamp){
                std::getline(fin_timestamp, line_timestamp);
                ss_timestamp = std::stringstream(line_timestamp);
                while (std::getline(ss_timestamp, item_timestamp, ' '))
                    stream_timestamp_origin.push_back(item_timestamp);
            }
            
            int i = 0;
            while (std::getline(ss, item, ' ')) {
                if (counts.at(item) >= frequency_min_count_) {
                    if (!use_pretrained_keys)
                        stream.push_back(item);
                    else if (idmap_.find(item) != idmap_.end())
                        stream.push_back(item);
                    
                    if (use_timestamp)
                        stream_timestamp.push_back(stream_timestamp_origin.at(i));
                }
                i++;
            }
            int size = static_cast<int>(stream.size());
            if (size < consume_min_count_ + 1)
                continue;
            if (max_samples_per_stream_ > 0) {
                offset = std::max(0, size - 1 - sequence_length_ - (max_samples_per_stream_ - 1));
                stream.erase(stream.begin(), stream.begin() + offset);
                if (use_timestamp)
                    stream_timestamp.erase(stream_timestamp.begin(), stream_timestamp.begin() + offset);
            }
            if (!use_pretrained_keys) {
                for (const auto& item : stream) {
                    if (idmap_.find(item) == idmap_.end())
                        idmap_[item] = static_cast<int>(idmap_.size());
                }
            }
            fout << __join(stream) << '\n';
            if (use_timestamp)
                fout_timestamp << __join(stream_timestamp) << '\n';
            ++num_lines;
        }
        fout.close();
        fin.close();
        if (use_timestamp){
            fout_timestamp.close();
            fin_timestamp.close();
        }

        if (num_lines == 0)
            throw std::runtime_error("No data in given file.");

        // Forth, split files into several chunks.
        int q = num_lines / num_workers_, r = num_lines % num_workers_;
        std::vector<int> num_lines_per_worker(num_workers_, q);
        if (r != 0) {
            for (int _r = 0; _r < r; ++_r)
                ++num_lines_per_worker[_r];
        }

        if (q == 0) {
            num_lines_per_worker.erase(num_lines_per_worker.begin() + r, num_lines_per_worker.end());
            num_workers_ = static_cast<int>(num_lines_per_worker.size());
        }

        int count = 0, worker_id = 0;
        std::string chunk_fname, chunk_fname_timestamp;
        fin = std::ifstream(fname_ + ".tmp");
        if (use_timestamp)
            fin_timestamp = std::ifstream(fname_timestamp_ + ".tmp");
        while (std::getline(fin, line)) {
            if (use_timestamp)
                std::getline(fin_timestamp, line_timestamp);
            if (count == 0) {
                chunk_fname = tmp_dir_ + "/" + "chunk_" + std::to_string(worker_id);
                fout = std::ofstream(chunk_fname);
                chunk_fnames_.push_back(chunk_fname);
                if (use_timestamp){
                    chunk_fname_timestamp = tmp_dir_ + "/" + "timestamp_chunk_" + std::to_string(worker_id);
                    fout_timestamp = std::ofstream(chunk_fname_timestamp);
                    chunk_fnames_timestamp_.push_back(chunk_fname_timestamp);
                }
            }
            fout << line << '\n';
            if (use_timestamp)
                fout_timestamp << line_timestamp << '\n';
            ++count;
            if (count == num_lines_per_worker[worker_id]) {
                count = 0;
                ++worker_id;
                fout.close();
                if (use_timestamp)
                    fout_timestamp.close();
            }
        }
        fin.close();
        std::remove((fname_ + ".tmp").c_str());
        if (use_timestamp){
            fout_timestamp.close();
            std::remove((fname_timestamp_ + ".tmp").c_str());
        }
    }

    void __process_by_chunks() {
        
        std::vector<std::thread> workers;
        workers.reserve(num_workers_ + num_workers_);
        for (int worker_id = 0; worker_id < num_workers_; ++worker_id) {
            workers.push_back(std::thread(
                [this, worker_id] () {
                    this->__process_by_chunks_job(this->chunk_fnames_[worker_id], 0);
            }));
            workers.push_back(std::thread(
                [this, worker_id] () {
                    bool use_timestamp = !this->fname_timestamp_.empty();
                    if (use_timestamp)
                        this->__process_by_chunks_job(this->chunk_fnames_timestamp_[worker_id], 1);
            }));
        }
        // Join threads.
        for (auto&& w : workers)
            w.join();
    }

    void __process_by_chunks_job(std::string& chunk_fname, int mode){
        std::ifstream fin(chunk_fname);
        std::ofstream fout_train(chunk_fname + "_train", std::ios::binary), fout_vali(chunk_fname + "_vali", std::ios::binary);
        std::stringstream ss;
        std::string line, item;
        int pad_token = static_cast<int>(this->idmap_.size()) + 1;
        if(mode == 1)
            pad_token = 0;
        std::vector<int> buf(this->sequence_length_);
        while (std::getline(fin, line)) {
            ss = std::stringstream(line);
            std::vector<int> stream;
            while (std::getline(ss, item, ' ')){
                if(mode == 0)
                    stream.push_back(idmap_[item]);
                else
                    stream.push_back(stoi(item.substr(0, item.size()-3)));
            }

            int size = static_cast<int>(stream.size());
            int beg, end;
            int num_iter = std::max(0, size - 1 - this->sequence_length_) + 1;
            for (int i = 0; i < num_iter; ++i) {
                beg = std::max(0, size - 1 - i - this->sequence_length_);
                end = size - 1 - i;
                int left = this->sequence_length_ - (end - beg);
                std::fill(buf.begin(), buf.begin() + left, pad_token);
                std::copy(stream.cbegin() + beg, stream.cbegin() + end, buf.begin() + left);  // Note: pad_mask = num_items + 1
                fout_train.write(reinterpret_cast<char*>(buf.data()), sizeof(int) * this->sequence_length_);
                if (i == 0) {
                    fout_vali.write(reinterpret_cast<char*>(std::next(buf.data())), sizeof(int) * (this->sequence_length_ - 1));
                    fout_vali.write(reinterpret_cast<char*>(&stream.back()), sizeof(int));
                }
            }


            num_iter = std::max(0, size - this->sequence_length_) + 1;
            for (int i = 0; i < num_iter; ++i) {
                beg = std::max(0, size - i - this->sequence_length_);
                end = size - i;
                int left = this->sequence_length_ - (end - beg);
                std::fill(buf.begin(), buf.begin() + left, pad_token);
                std::copy(stream.cbegin() + beg, stream.cbegin() + end, buf.begin() + left);  // Note: pad_mask = num_items + 1
                fout_train.write(reinterpret_cast<char*>(buf.data()), sizeof(int) * this->sequence_length_);
                if (i == 0) {
                    fout_vali.write(reinterpret_cast<char*>(std::next(buf.data())), sizeof(int) * (this->sequence_length_ - 1));
                    fout_vali.write(reinterpret_cast<char*>(&stream.back()), sizeof(int));
                }
            }
        }
        fin.close();
        fout_train.close();
        fout_vali.close();
        std::remove(chunk_fname.c_str());
    }

    void __aggregate(const std::string& suffix, std::vector<std::string>& chunk_fnames_ref, const std::string& output_file_suffix) {
        std::ofstream fout(tmp_dir_ + "/" + suffix + output_file_suffix, std::ios::binary);
        std::ifstream fin;
        std::string fname;
        std::vector<char> buf(8192);
        for (const auto& chunk_fname : chunk_fnames_ref) {
            fname = chunk_fname + "_" + suffix;
            fin = std::ifstream(fname, std::ios::binary);
            while (!fin.eof()) {
                fin.read(buf.data(), buf.size());
                fout.write(buf.data(), fin.gcount());
            }
            fin.close();
            std::remove(fname.c_str());
        }
        fout.close();
    }

    void __write_keys() {
        std::vector<std::string> keys;
        keys.reserve(idmap_.size());
        for (const auto& kv : idmap_)
            keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end(), [this] (const auto& left, const auto& right) { return this->idmap_[left] < this->idmap_[right]; });
        std::ofstream fout(tmp_dir_ + "/" + "keys");
        for (const auto& key : keys)
            fout << key << '\n';
    }

    void __map() {
        __split_chunks();
        __process_by_chunks();
    }

    void __reduce() {
        __aggregate("train", chunk_fnames_, "");
        __aggregate("vali", chunk_fnames_, "");
        bool use_timestamp = !fname_timestamp_.empty();

        if (use_timestamp){
            __aggregate("train", chunk_fnames_timestamp_, "_timestamp");
            __aggregate("vali", chunk_fnames_timestamp_, "_timestamp");
        }
        __write_keys();
    }

 private:
    std::vector<std::string> chunk_fnames_, chunk_fnames_timestamp_;
    std::unordered_map<std::string, int> idmap_;
    const std::string fname_;
    const std::string fname_timestamp_;
    const std::string tmp_dir_;
    const std::string pretrained_key_fname_;
    const int frequency_min_count_;
    const int consume_min_count_;
    const int sequence_length_;
    const int max_samples_per_stream_;
    int num_workers_;
    bool use_timestamp;
};

