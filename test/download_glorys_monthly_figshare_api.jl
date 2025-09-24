using Downloads, JSON3, HTTP, SHA

figshare_id = 30193228

function get_figshare_download_url(article_id)
    api_url = "https://api.figshare.com/v2/articles/$article_id"
    
    try
        response = HTTP.get(api_url)
        if response.status == 200
            data = JSON3.read(String(response.body))
            return data.files
        else
            @warn "API request failed with status: $(response.status)"
            return nothing
        end
    catch e
        @warn "Error accessing Figshare API: $e"
        return nothing
    end
end

"""
    generate_cache_key(files_info)

Generate a hash-based cache key from file metadata for GitHub Actions caching.
"""
function generate_cache_key(files_info)
    # Create a string representation of file metadata
    metadata = join(["$(file.name):$(file.size)" for file in files_info], ";")
    # Generate a hash for the cache key
    return bytes2hex(sha256(metadata))
end

"""
    check_file_integrity(filepath, expected_size)

Check if a file exists and has the expected size.
"""
function check_file_integrity(filepath, expected_size)
    if !isfile(filepath)
        return false
    end
    
    file_size = filesize(filepath)
    if file_size != expected_size
        @warn "File size mismatch for $filepath: expected $expected_size, got $file_size"
        return false
    end
    
    return true
end

"""
    download_from_figshare_api(article_id::Int, download_dir::String=".")

Downloads files from a Figshare article using the official API method.
Implements caching to avoid re-downloading files that already exist.
"""
function download_from_figshare_api(article_id, download_dir::String=".", force_download::Bool=false)
    files_info = get_figshare_download_url(article_id)
    
    if files_info === nothing
        @error "Could not get file information for article $article_id"
        return false
    end
    
    # Create cache key file for GitHub Actions
    cache_key = generate_cache_key(files_info)
    println("Cache key: $cache_key")
    
    if !isdir(download_dir)
        mkpath(download_dir)
    end
    
    # Write cache key to a file
    cache_key_file = joinpath(download_dir, "cache_key.txt")
    open(cache_key_file, "w") do io
        println(io, cache_key)
    end
    
    # Check if all files already exist
    if !force_download && check_all_files_exist(files_info, download_dir)
        println("✓ All files already exist with correct sizes. Skipping downloads.")
        return true
    end
    
    success_count = 0
    cached_count = 0
    
    for file in files_info
        download_url = file.download_url
        filename = file.name
        filepath = joinpath(download_dir, filename)
        expected_size = file.size
        
        # Check if file already exists with correct size
        if check_file_integrity(filepath, expected_size)
            println("✓ Using cached file: $filename")
            cached_count += 1
            success_count += 1
            continue
        end
        
        println("Downloading: $filename ($expected_size bytes)")
        
        # Use proper headers to avoid being blocked
        headers = [
            "User-Agent" => "Julia Scientific Computing Client/1.0",
            "Accept" => "*/*",
            "Referer" => "https://figshare.com"
        ]
        
        try
            Downloads.download(download_url, filepath; headers=headers, timeout=120)
            if check_file_integrity(filepath, expected_size)
                println("✓ Successfully downloaded: $filename")
                success_count += 1
            else
                @warn "Downloaded file has incorrect size: $filename"
            end
        catch e
            @warn "Failed to download $filename: $e"
        end
    end
    
    println("Files: $success_count/$(length(files_info)) available ($(cached_count) from cache)")
    return success_count > 0
end

"""
    check_all_files_exist(files_info, download_dir)

Check if all files from files_info exist in download_dir with correct sizes.
"""
function check_all_files_exist(files_info, download_dir::String=".")
    all_files_exist = true
    for file in files_info
        filepath = joinpath(download_dir, file.name)
        if !check_file_integrity(filepath, file.size)
            all_files_exist = false
            break
        end
    end
    return all_files_exist
end

# Only run if executed directly, not when included
if abspath(PROGRAM_FILE) == @__FILE__
    download_from_figshare_api(figshare_id, "./GLORYS_data")
end