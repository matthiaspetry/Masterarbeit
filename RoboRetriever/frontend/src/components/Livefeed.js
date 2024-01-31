const videoStreamUrl = "http://localhost:5000/video_feed";

const Livefeed = () => {
  return (
    <div>
      <h2 className=" text-lg font-semibold mb-4 ">Live Feed</h2>
      <div className="relative" style={{ maxWidth: '800px' }}>
        {/* Adjust the maxWidth to your desired width */}
        <img
          src={videoStreamUrl}
          alt="Live Feed"
          style={{ width: '100%', height: '100' }}
        />
      </div>
    </div>
  );
};

export default Livefeed;