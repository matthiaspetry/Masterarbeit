import VideoCard from "@/components/videocard";
import Control from "@/components/Control";
import Status from "@/components/Status";


export default function Home() {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: '20px', padding: '20px' }}>
      <div style={{ flex: '0 0 800px' }}> {/* Adjust the flex-basis to your preference */}
        <VideoCard />
      </div>
      <div style={{ flex: '0 0 800px' }}> {/* Adjust the flex-basis to your preference */}
       <Control/>
      </div>
      <div style={{ flex: '0 0 800px' }}> {/* Adjust the flex-basis to your preference */}
      <Status></Status>
      </div>

      
    </div>
  );
}