import Livefeed from "@/components/Livefeed";
import Control from "@/components/Control";
import Status from "@/components/Status";
import Chart from "@/components/Chart";
import Stats from "@/components/Stats";

import { Card } from "@tremor/react";


export default function Home() {
  return (
    <main className="pt-8 pl-8 pr-8" >
      <div className="grid grid-cols-5 grid-rows-6 gap-4 h-full ">
    <div className="col-span-2 row-span-3 h-full">
      <Card className="h-full">
        <Livefeed/>
      </Card>
    </div>
    <div className="col-span-2 row-span-3 col-start-3">

    <Card className="h-full">
    <Control/>
      </Card>
     
    </div>
    <div className="row-span-3 col-start-5">
    <Card className="h-full">
    <Status/>
      </Card>
      
    </div>
    <div className="col-span-3 row-span-3 col-start-3 row-start-4">
    <Card className="h-full">
    <Chart/>
    </Card>
    </div>
    <div className="col-span-2 row-span-3 col-start-1 row-start-4">
    <Card className="h-full">
    <Stats/>
      </Card>
    </div>
</div>
 
    </main>
  );
}
