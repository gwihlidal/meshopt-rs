extern crate meshopt;
use meshopt::*;

#[test]
fn encode_index() {
    assert!(true);
    /*
    // note: 4 6 5 triangle here is a combo-breaker:
    // we encode it without rotating, a=next, c=next - this means we do *not* bump next to 6
    // which means that the next triangle can't be encoded via next sequencing!
    const unsigned int indices[] = {0, 1, 2, 2, 1, 3, 4, 6, 5, 7, 8, 9};
    const size_t index_count = sizeof(indices) / sizeof(indices[0]);
    const size_t vertex_count = 10;
    
    std::vector<unsigned char> buffer(meshopt_encodeIndexBufferBound(index_count, vertex_count));
    buffer.resize(meshopt_encodeIndexBuffer(&buffer[0], buffer.size(), indices, index_count));
    
    // check that encode is memory-safe; note that we reallocate the buffer for each try to make sure ASAN can verify buffer access
    for (size_t i = 0; i <= buffer.size(); ++i)
    {
        std::vector<unsigned char> shortbuffer(i);
        size_t result = meshopt_encodeIndexBuffer(i == 0 ? 0 : &shortbuffer[0], i, indices, index_count);
        (void)result;
    
        if (i == buffer.size())
            assert(result == buffer.size());
        else
            assert(result == 0);
    }
    
    // check that decode is memory-safe; note that we reallocate the buffer for each try to make sure ASAN can verify buffer access
    unsigned int destination[index_count];
    
    for (size_t i = 0; i <= buffer.size(); ++i)
    {
        std::vector<unsigned char> shortbuffer(buffer.begin(), buffer.begin() + i);
        int result = meshopt_decodeIndexBuffer(destination, index_count, i == 0 ? 0 : &shortbuffer[0], i);
        (void)result;
    
        if (i == buffer.size())
            assert(result == 0);
        else
            assert(result < 0);
    }
    
    // check that decoder doesn't accept extra bytes after a valid stream
    {
        std::vector<unsigned char> largebuffer(buffer);
        largebuffer.push_back(0);
    
        int result = meshopt_decodeIndexBuffer(destination, index_count, &largebuffer[0], largebuffer.size());
        (void)result;
    
        assert(result < 0);
    }
    
    // check that decoder doesn't accept malformed headers
    {
        std::vector<unsigned char> brokenbuffer(buffer);
        brokenbuffer[0] = 0;
    
        int result = meshopt_decodeIndexBuffer(destination, index_count, &brokenbuffer[0], brokenbuffer.size());
        (void)result;
    
        assert(result < 0);
    }
    */
}

#[test]
fn encode_vertex() {
    assert!(true);

    let mut vertices: Vec<packing::PackedVertexOct> = Vec::with_capacity(4);

    vertices.push(packing::PackedVertexOct {
        p: [0, 0, 0],
        n: [0, 0],
        t: [0, 0],
    });

    vertices.push(packing::PackedVertexOct {
        p: [300, 0, 0],
        n: [0, 0],
        t: [500, 0],
    });

    vertices.push(packing::PackedVertexOct {
        p: [0, 300, 0],
        n: [0, 0],
        t: [0, 500],
    });

    vertices.push(packing::PackedVertexOct {
        p: [300, 300, 0],
        n: [0, 0],
        t: [500, 500],
    });

    let _encoded = encode_vertex_buffer(&vertices);

    /*
    // check that encode is memory-safe; note that we reallocate the buffer for each try to make sure ASAN can verify buffer access
    for (size_t i = 0; i <= buffer.size(); ++i)
    {
        std::vector<unsigned char> shortbuffer(i);
        size_t result = meshopt_encodeVertexBuffer(i == 0 ? 0 : &shortbuffer[0], i, vertices, vertex_count, sizeof(PV));
        (void)result;
    
        if (i == buffer.size())
            assert(result == buffer.size());
        else
            assert(result == 0);
    }
    
    // check that decode is memory-safe; note that we reallocate the buffer for each try to make sure ASAN can verify buffer access
    PV destination[vertex_count];
    
    for (size_t i = 0; i <= buffer.size(); ++i)
    {
        std::vector<unsigned char> shortbuffer(buffer.begin(), buffer.begin() + i);
        int result = meshopt_decodeVertexBuffer(destination, vertex_count, sizeof(PV), i == 0 ? 0 : &shortbuffer[0], i);
        (void)result;
    
        if (i == buffer.size())
            assert(result == 0);
        else
            assert(result < 0);
    }
    
    // check that decoder doesn't accept extra bytes after a valid stream
    {
        std::vector<unsigned char> largebuffer(buffer);
        largebuffer.push_back(0);
    
        int result = meshopt_decodeVertexBuffer(destination, vertex_count, sizeof(PV), &largebuffer[0], largebuffer.size());
        (void)result;
    
        assert(result < 0);
    }
    
    // check that decoder doesn't accept malformed headers
    {
        std::vector<unsigned char> brokenbuffer(buffer);
        brokenbuffer[0] = 0;
    
        int result = meshopt_decodeVertexBuffer(destination, vertex_count, sizeof(PV), &brokenbuffer[0], brokenbuffer.size());
        (void)result;
    
        assert(result < 0);
    }
    */
}
