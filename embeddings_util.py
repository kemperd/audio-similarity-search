from pymilvus import MilvusClient, DataType
import librosa
import torch
import numpy as np

SAMPLE_RATE = 16000

# Milvus
MILVUS_DATABASE = 'esc50.db'
MILVUS_COLLECTION_NAME = 'esc50'
EMBEDDINGS_DIMENSIONS = 32768

    ####### BACKUP BLOCK

    #feature_vector = torch.mean(output.extract_features.squeeze(), 0).numpy()
    #return feature_vector
    #return minmax_scale(feature_vector, feature_range=(0,1))

    # length of video_features depends on the length of the video, containing 512 elements each
    # get the features corresponding to the middle of the video
    #video_features = output.extract_features.squeeze().numpy()
    #video_features = video_features[len(video_features)//2]
    #return minmax_scale(video_features, feature_range=(0,1))


    #feature_vector = np.concatenate(output.extract_features.squeeze().numpy()[0],
    #                                output.extract_features.squeeze().numpy()[100])
    #return feature_vector

    #return [ output.extract_features.squeeze().numpy()[0], output.extract_features.squeeze().numpy()[100]]

    #return np.stack([output.extract_features.squeeze().numpy()[0], output.extract_features.squeeze().numpy()[100]], axis=1)

    # based on minmax-scale, different from normalization?
    #return minmax_scale(feature_vector, feature_range=(0,1))

    ####### END BACKUP BLOCK


def retrieve_embeddings_for_audiofile(filename, feature_extractor, model):
    input_audio, sample_rate = librosa.load(filename, sr=SAMPLE_RATE, mono=True)
    input_features = feature_extractor(
        input_audio, 
        return_tensors='pt', 
        sampling_rate=SAMPLE_RATE
    )
    with torch.no_grad():
        output = model(input_features.input_values)

    file_length = len(output.extract_features.squeeze().numpy()) - 1

    # Maximum Milvus dimensions is 32768. 
    # Wav2Vec2 feature vector is 512, so we can fit 64 feature vectors into Milvus
    indexes = np.linspace(0, file_length, 64, dtype=int)

    feature_list = []
    for index in indexes:
        feature_list.append(output.extract_features.squeeze().numpy()[index])

    feature_vector = np.concatenate(feature_list)
    return feature_vector/np.linalg.norm(feature_vector)


def insert_embeddings_into_db(feature_vector, filename, milvus_client):
    data = [ { 'filename': filename, 'embeddings': feature_vector } ]
    milvus_client.insert(collection_name=MILVUS_COLLECTION_NAME, data=data)

def init_milvus(milvus_client):
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name='filename', max_length=500, datatype=DataType.VARCHAR)
    schema.add_field(field_name='embeddings', datatype=DataType.FLOAT_VECTOR, 
                     dim=EMBEDDINGS_DIMENSIONS, description='sample embeddings vector')

    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="embeddings",
        index_type="AUTOINDEX",
        metric_type="IP",
    )

    milvus_client.create_collection(
    collection_name=MILVUS_COLLECTION_NAME,
    schema=schema,
    index_params=index_params,
)    
    
def retrieve_by_id(id, milvus_client):
    # Search collection by ID

    res_by_id = milvus_client.query(
        collection_name=MILVUS_COLLECTION_NAME,
        filter=f"id == {id}"
    )
    return res_by_id    

def retrieve_by_sample(feature_vector, milvus_client):
    # # Perform a vector search for similar texts
    return milvus_client.search(
        collection_name=MILVUS_COLLECTION_NAME,
        data=[feature_vector],  # The query embedding
        anns_field="embeddings",  # Field that contains the embeddings
    #    param={"metric_type": "COSINE", "params": {"ef": 128}},  # Search parameters
        limit=20,  # Number of results to return
    #    expr=None  # Optional filtering expression
        search_params={"metric_type": "IP"},
        output_fields=['filename'],
    )
