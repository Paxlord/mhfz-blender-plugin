import collections

class AdjTriangle:
    def __init__(self, v0, v1, v2, face_index):
        self.v_ref = (v0, v1, v2)
        self.face_index = face_index
        self.adj_tri = [None, None, None]
    def __repr__(self):
        return f"Face {self.face_index}: V({self.v_ref}), Adj({self.adj_tri})"
    def find_edge_index(self, v_a, v_b):
        v0, v1, v2 = self.v_ref
        if (v_a == v0 and v_b == v1) or (v_a == v1 and v_b == v0): return 0
        if (v_a == v0 and v_b == v2) or (v_a == v2 and v_b == v0): return 1
        if (v_a == v1 and v_b == v2) or (v_a == v2 and v_b == v1): return 2
        raise ValueError(f"Edge ({v_a}, {v_b}) not found in triangle {self.v_ref}")
    def opposite_vertex(self, v_a, v_b):
        for v in self.v_ref:
            if v != v_a and v != v_b:
                return v
        raise ValueError(f"Edge ({v_a}, {v_b}) not found in triangle {self.v_ref}")
    def connectivity(self):
        return sum(1 for link in self.adj_tri if link is not None)

class Striper:
    DEBUG = False 

    def __init__(self):
        self.faces = None
        self.adjacency_data = None
        self.options = {}

    def log(self, message):
        if self.DEBUG:
            print(f"[Striper] {message}")

    def _build_adjacency(self):
        adj_faces = [AdjTriangle(v0, v1, v2, i) for i, (v0, v1, v2) in enumerate(self.faces)]
        edge_map = {}
        for i, face_v in enumerate(self.faces):
            v0, v1, v2 = face_v
            edges = [(tuple(sorted((v0, v1))), 0), (tuple(sorted((v0, v2))), 1), (tuple(sorted((v1, v2))), 2)]
            for edge_key, local_edge_index in edges:
                if edge_key not in edge_map: edge_map[edge_key] = []
                edge_map[edge_key].append((i, local_edge_index))

        for edge_key, face_pairs in edge_map.items():
            if len(face_pairs) == 2:
                (f1_idx, e1_idx), (f2_idx, e2_idx) = face_pairs
                adj_faces[f1_idx].adj_tri[e1_idx] = (f2_idx, e2_idx)
                adj_faces[f2_idx].adj_tri[e2_idx] = (f1_idx, e1_idx)
        self.adjacency_data = adj_faces

    def _track_strip(self, start_face_index, v_oldest, v_middle, visited_faces):
        self.log(f"  _track_strip starting at face {start_face_index} with edge ({v_oldest}, {v_middle})")
        
        strip_v, strip_f = [v_oldest, v_middle], []
        curr_f_idx = start_face_index
        
        faces_in_current_track = set()

        while curr_f_idx is not None:
            if curr_f_idx in faces_in_current_track:
                self.log(f"    Cycle detected! Tried to re-visit face {curr_f_idx}. Terminating track.")
                break
            faces_in_current_track.add(curr_f_idx)
            
            self.log(f"    Tracking through face {curr_f_idx}")
            
            curr_f = self.adjacency_data[curr_f_idx]
            v_newest = curr_f.opposite_vertex(v_oldest, v_middle)
            strip_v.append(v_newest)
            strip_f.append(curr_f_idx)
            edge_idx = curr_f.find_edge_index(v_middle, v_newest)
            link = curr_f.adj_tri[edge_idx]
            v_oldest, v_middle = v_middle, v_newest
            
            if link is None:
                self.log(f"    Reached boundary edge. Terminating track.")
                break
            
            next_f_idx, _ = link
            if next_f_idx in visited_faces:
                self.log(f"    Next face {next_f_idx} is already in a global strip. Terminating track.")
                break
            
            curr_f_idx = next_f_idx
            
        return strip_v, strip_f

    def _compute_best_strip(self, start_face_index, visited_faces):
        self.log(f" _compute_best_strip for start_face {start_face_index}")
        start_face = self.adjacency_data[start_face_index]
        v0, v1, v2 = start_face.v_ref
        start_edges = [(v0, v1), (v2, v0), (v1, v2)]
        best_strip_v, best_strip_f, best_forward_len = [], [], 0

        for i, (v_start, v_end) in enumerate(start_edges):
            self.log(f"  Direction {i+1}/3: Edge ({v_start}, {v_end})")
            local_visited = visited_faces.copy()
            forward_v, forward_f = self._track_strip(start_face_index, v_start, v_end, local_visited)
            for f_idx in forward_f: local_visited.add(f_idx)
            
            forward_len = len(forward_v)
            forward_v.reverse()
            forward_f.reverse()
            
            v_new_start, v_new_end = forward_v[-2], forward_v[-1]
            start_face_bw = self.adjacency_data[start_face_index]
            entry_edge_idx = start_face_bw.find_edge_index(v_new_start, v_new_end)
            link = start_face_bw.adj_tri[entry_edge_idx]
            backward_v, backward_f = [], []
            if link:
                neighbor_f_idx, _ = link
                if neighbor_f_idx not in local_visited:
                    backward_v_full, backward_f = self._track_strip(neighbor_f_idx, v_new_start, v_new_end, local_visited)
                    backward_v = backward_v_full[2:]
            
            combined_v = forward_v + backward_v
            combined_f = forward_f + backward_f

            if len(combined_v) > len(best_strip_v):
                best_strip_v, best_strip_f, best_forward_len = combined_v, combined_f, forward_len

        if self.options['OneSided'] and best_forward_len % 2 == 1:
            best_strip_v.reverse()
            new_pos = len(best_strip_v) - best_forward_len
            if new_pos % 2 == 0:
                best_strip_v.insert(0, best_strip_v[0])

        self.log(f"  Best strip found for start_face {start_face_index} has length {len(best_strip_v)}")
        return best_strip_v, best_strip_f
    
    def _connect_all_strips(self, strips):
        if not strips or len(strips) <= 1:
            return strips
        combined_strip = strips[0][:]
        for i in range(1, len(strips)):
            next_strip = strips[i][:]
            last_vert, first_vert = combined_strip[-1], next_strip[0]
            combined_strip.extend([last_vert, first_vert])
            if self.options['OneSided'] and len(combined_strip) % 2 == 1:
                if len(next_strip) > 1 and next_strip[0] != next_strip[1]:
                    combined_strip.append(first_vert)
                else:
                    next_strip.pop(0)
            combined_strip.extend(next_strip)
        return [combined_strip]

    def stripify(self, faces, sgi_algorithm=True, one_sided=True, connect_all_strips=False):
        if not faces:
            return []
            
        self.faces = faces
        self.options = {'SGIAlgorithm': sgi_algorithm, 'OneSided': one_sided, 'ConnectAllStrips': connect_all_strips}
        
        self.log(f"Starting stripification for {len(faces)} faces.")
        
        self._build_adjacency()
        
        num_faces = len(self.faces)
        insertion_order = list(range(num_faces))
        if self.options['SGIAlgorithm']:
            insertion_order.sort(key=lambda i: self.adjacency_data[i].connectivity())

        all_strips = []
        visited_faces = set()
        
        processed_faces = 0
        for face_index in insertion_order:
            if face_index not in visited_faces:
                self.log(f"Starting new strip from face {face_index} (connectivity: {self.adjacency_data[face_index].connectivity()})")
                best_v, best_f = self._compute_best_strip(face_index, visited_faces)
                
                for f_idx in best_f:
                    visited_faces.add(f_idx)
                
                all_strips.append(best_v)
                processed_faces += len(best_f)
                self.log(f"Created strip of length {len(best_v)} using {len(best_f)} faces. Total faces processed: {processed_faces}/{num_faces}")


        self.log(f"Finished. Found {len(all_strips)} strips.")
        if self.options['ConnectAllStrips']:
            self.log("Connecting all strips into one.")
            return self._connect_all_strips(all_strips)

        return all_strips